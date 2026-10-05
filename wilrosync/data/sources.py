"""Dataset sources: HDTF and TalkVid (YouTube segments from their official metadata) and MEAD (local).

Each source yields *segments*: ``{"id", "source", "url", "start", "end", "group"}`` where ``group`` is the
identity used for held-out splits and per-person caps (HDTF video name, TalkVid Person ID).
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import random
import subprocess

HDTF_REPO = "https://github.com/MRzzm/HDTF.git"
TALKVID_REPO = "FreedomIntelligence/TalkVid"
TALKVID_META = "data/filtered_video_clips.json"


def _ts(s: str) -> float:
    """'01:35' / '1:02:03' / '95' -> seconds."""
    out = 0.0
    for part in s.strip().split(":"):
        out = out * 60 + float(part)
    return out


def stable_fraction(key: str) -> float:
    """Deterministic pseudo-random number in [0, 1) for ``key`` (used for held-out splits)."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 2**32


# ----------------------------------------------------------------------------- HDTF
def fetch_hdtf_annotations(dest: str) -> str:
    """Clone the HDTF repo (annotations only, ~5 MB) once; returns the HDTF_dataset folder."""
    ann = os.path.join(dest, "HDTF_dataset")
    if not os.path.isdir(ann):
        subprocess.run(["git", "clone", "-q", "--depth", "1", HDTF_REPO, dest], check=True)
    return ann


def hdtf_segments(ann_dir: str) -> list[dict]:
    """Talking-head segments from the HDTF annotation files (xx_video_url.txt + xx_annotion_time.txt)."""
    segs = []
    for url_file in sorted(glob.glob(os.path.join(ann_dir, "*_video_url.txt"))):
        prefix = os.path.basename(url_file).split("_")[0]
        urls = {}
        for line in open(url_file):
            parts = line.split()
            if len(parts) >= 2:
                urls[parts[0].removesuffix(".mp4")] = parts[1]
        time_file = os.path.join(ann_dir, f"{prefix}_annotion_time.txt")
        if not os.path.isfile(time_file):
            continue
        for line in open(time_file):
            parts = line.split()
            if not parts:
                continue
            name = parts[0].removesuffix(".mp4")
            if name not in urls:
                continue
            for i, rng in enumerate(parts[1:]):
                if "-" not in rng:
                    continue
                a, b = rng.split("-", 1)
                start, end = _ts(a), _ts(b)
                if end > start:
                    segs.append({"id": f"hdtf_{name}_{i}", "source": "hdtf", "url": urls[name],
                                 "start": start, "end": end, "group": name})
    return segs


# ----------------------------------------------------------------------------- TalkVid
def fetch_talkvid_metadata(cache_dir: str | None = None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(TALKVID_REPO, TALKVID_META, repo_type="dataset", cache_dir=cache_dir)


def talkvid_segments(meta_path: str, min_dover: float = 8.5, min_orientation: float = 85.0,
                     min_rotation: float = 75.0, min_face_res: float = 150.0, languages: list[str] | None = None,
                     min_seconds: float = 3.0) -> list[dict]:
    """TalkVid clips passing the quality filters (scores shipped with the dataset).

    Defaults keep roughly the better half: DOVER >= 8.5 (median), near-frontal heads (orientation/rotation),
    and faces with ``avg_resolution`` >= 150 px."""
    with open(meta_path) as f:
        rows = json.load(f)
    langs = set(languages) if languages else None
    segs = []
    for r in rows:
        hd = r.get("head_detail", {}).get("scores")
        if hd is None or "dover_scores" not in r:
            continue
        if (r["dover_scores"] < min_dover or hd["min_orientation"] < min_orientation
                or hd["min_rotation"] < min_rotation or hd["avg_resolution"] < min_face_res):
            continue
        if langs and r["info"].get("Language") not in langs:
            continue
        start, end = float(r["start-time"]), float(r["end-time"])
        if end - start < min_seconds:
            continue
        segs.append({"id": f"talkvid_{r['id']}", "source": "talkvid", "url": r["info"]["Video Link"],
                     "start": start, "end": end, "group": f"talkvid_{r['info']['Person ID']}",
                     "lang": r["info"].get("Language"), "dover": r["dover_scores"]})
    return segs


# ----------------------------------------------------------------------------- selection
def split_holdout(segs: list[dict], fraction: float = 0.0, groups: int = 0) -> tuple[list[dict], list[dict]]:
    """Hold out whole identities: a deterministic ``fraction`` of groups, or the first ``groups`` groups by hash."""
    names = sorted({s["group"] for s in segs}, key=stable_fraction)
    held = set(names[:groups]) if groups else {g for g in names if stable_fraction(g) < fraction}
    return [s for s in segs if s["group"] not in held], [s for s in segs if s["group"] in held]


def select_budget(segs: list[dict], hours: float | None = None, max_segments: int | None = None,
                  max_seconds_per_group: float | None = None, seed: int = 0) -> list[dict]:
    """Pick segments up to ``hours`` of source video, spreading over identities (round-robin over groups,
    at most ``max_seconds_per_group`` per identity)."""
    rng = random.Random(seed)
    by_group: dict[str, list[dict]] = {}
    for s in segs:
        by_group.setdefault(s["group"], []).append(s)
    for g in by_group.values():
        rng.shuffle(g)
    order = list(by_group)
    rng.shuffle(order)
    budget = hours * 3600 if hours else float("inf")
    used, per_group, out = 0.0, dict.fromkeys(order, 0.0), []
    for rnd in itertools.count():
        progressed = False
        for g in order:
            if rnd >= len(by_group[g]):
                continue
            s = by_group[g][rnd]
            d = s["end"] - s["start"]
            if max_seconds_per_group and per_group[g] + d > max_seconds_per_group:
                continue
            if used + d > budget or (max_segments and len(out) >= max_segments):
                return out
            out.append(s)
            used += d
            per_group[g] += d
            progressed = True
        if not progressed:
            return out


def total_hours(segs: list[dict]) -> float:
    return sum(s["end"] - s["start"] for s in segs) / 3600


# ----------------------------------------------------------------------------- MEAD (local)
def mead_pairs(mead_root: str, views: tuple[str, ...] = ("front",), per_folder: int = 4,
               seed: int = 0, relative_to: str | None = None) -> list[dict]:
    """Pseudo pairs from an extracted MEAD tree (``<actor>/video/<view>/<emotion>/<level>/<nnn>.mp4``):
    two different sentences with the same actor, view, emotion and level."""
    rng = random.Random(seed)
    rows = []
    for view in views:
        for d in sorted(glob.glob(os.path.join(mead_root, "*", "video", view, "*", "*"))):
            clips = sorted(glob.glob(os.path.join(d, "*.mp4")))
            perms = list(itertools.permutations(clips, 2))
            for a, b in rng.sample(perms, k=min(per_folder, len(perms))):
                rel = (lambda p: os.path.relpath(p, relative_to)) if relative_to else (lambda p: p)
                rows.append({"kind": "pseudo_pair", "cond": rel(a), "target": rel(b),
                             "group": "mead_" + os.path.relpath(d, mead_root).split(os.sep)[0]})
    return rows
