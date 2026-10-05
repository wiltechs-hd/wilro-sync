"""Data stages (docs/DATASETS.md): smoke -> first_model -> paper_scale.

A stage YAML (``configs/data/<name>.yaml``) says which sources to use and how much of each. Running a stage:

1. **plan**     – list HDTF / TalkVid segments (official metadata), split held-out identities, apply the hour
                  budget, estimate storage. Saved to ``<work>/stages/<name>/plan.json``.
2. **acquire**  – download each segment (yt-dlp) and write a 512x512, 25 fps face crop to
                  ``<work>/raw/<source>/<id>.mp4``. Resumable; shared by all stages.
3. **manifest** – training manifest (+ MEAD pseudo pairs, + held-out list) for this stage.
4. **sync**     – SyncNet filter (``filter_manifest``); scores are cached and shared.
5. **prepare**  – latents / audio windows into the shared ``<work>/latents/clips`` folder, with a per-stage index.

Later stages reuse everything earlier stages downloaded, cropped, scored and prepared.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field

from omegaconf import OmegaConf

from ..pipeline.lipsync import DEFAULT_PROMPT
from . import sources

CLIP_MB_512_49 = 5.3  # prepared clip size at 512x512x49 frames
CROP_MB_PER_S = 0.35  # 512x512 H.264 face crop, crf 18
# mouth position / face width inside our face crops (face_crop.py: scale 1.8, y_shift 0.1)
CROP_MOUTH = [0.5, 0.58]
CROP_FACE_W = 0.55


@dataclass
class HDTFConfig:
    enabled: bool = True
    max_segments: int | None = None  # None = all
    holdout_groups: int = 20  # whole HDTF videos kept out of training (evaluation)


@dataclass
class TalkVidConfig:
    enabled: bool = False
    hours: float = 0.0
    max_seconds_per_person: float = 120.0
    min_dover: float = 8.5
    min_orientation: float = 85.0
    min_rotation: float = 75.0
    min_face_res: float = 150.0
    languages: list[str] | None = None
    holdout_fraction: float = 0.01


@dataclass
class MEADConfig:
    enabled: bool = False
    root: str = "mead"  # extracted MEAD tree, relative to <work>/raw (manual download)
    views: list[str] = field(default_factory=lambda: ["front"])
    per_folder: int = 4


@dataclass
class ClipConfig:
    size: int = 512
    frames: int = 49
    clips_per_minute: float = 3.0
    max_clips_per_video: int = 12
    caption: str = DEFAULT_PROMPT


@dataclass
class SyncConfig:
    min_conf: float = 3.0
    max_offset: int = 3


@dataclass
class StageConfig:
    name: str = "smoke"
    description: str = ""
    hdtf: HDTFConfig = field(default_factory=HDTFConfig)
    talkvid: TalkVidConfig = field(default_factory=TalkVidConfig)
    mead: MEADConfig = field(default_factory=MEADConfig)
    clips: ClipConfig = field(default_factory=ClipConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    extra_videos: list[str] = field(default_factory=list)  # folders under <work>/raw with your own face-cropped videos
    storage_budget_gb: float = 50.0
    train_overrides: dict = field(default_factory=dict)
    seed: int = 0


def stage_path(name_or_path: str) -> str:
    if os.path.isfile(name_or_path):
        return name_or_path
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (os.getcwd(), os.path.join(here, "..", "..")):
        p = os.path.join(root, "configs", "data", f"{name_or_path}.yaml")
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"no stage config {name_or_path!r} (configs/data/<name>.yaml)")


def load_stage(name_or_path: str, overrides: list[str] | None = None) -> StageConfig:
    cfg = OmegaConf.merge(OmegaConf.structured(StageConfig), OmegaConf.load(stage_path(name_or_path)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)


@dataclass
class StagePaths:
    work: str
    name: str

    @property
    def raw(self) -> str:
        return os.path.join(self.work, "raw")

    @property
    def stage(self) -> str:
        return os.path.join(self.work, "stages", self.name)

    @property
    def clips(self) -> str:
        return os.path.join(self.work, "latents", "clips")

    def p(self, *parts: str) -> str:
        return os.path.join(self.stage, *parts)


# ----------------------------------------------------------------------------- 1. plan
def plan_stage(cfg: StageConfig, work: str, talkvid_meta: str | None = None, hdtf_dir: str | None = None,
               force: bool = False) -> dict:
    paths = StagePaths(work, cfg.name)
    os.makedirs(paths.stage, exist_ok=True)
    train, held = [], []
    if cfg.hdtf.enabled:
        ann = hdtf_dir or sources.fetch_hdtf_annotations(os.path.join(work, "meta", "HDTF"))
        segs = sources.hdtf_segments(ann)
        tr, ho = sources.split_holdout(segs, groups=cfg.hdtf.holdout_groups)
        tr = sources.select_budget(tr, max_segments=cfg.hdtf.max_segments, seed=cfg.seed)
        train += tr
        held += ho
    if cfg.talkvid.enabled and cfg.talkvid.hours > 0:
        t = cfg.talkvid
        meta = talkvid_meta or sources.fetch_talkvid_metadata(os.path.join(work, "meta", "hf"))
        segs = sources.talkvid_segments(meta, t.min_dover, t.min_orientation, t.min_rotation, t.min_face_res,
                                        t.languages)
        tr, ho = sources.split_holdout(segs, fraction=t.holdout_fraction)
        train += sources.select_budget(tr, hours=t.hours, max_seconds_per_group=t.max_seconds_per_person,
                                       seed=cfg.seed)
        held += sources.select_budget(ho, max_segments=50, max_seconds_per_group=60, seed=cfg.seed)
    clips = sum(n_clips(s, cfg.clips) for s in train)
    scale = (cfg.clips.size / 512) ** 2 * (cfg.clips.frames / 49)
    est = {
        "train_segments": len(train),
        "train_hours": round(sources.total_hours(train), 2),
        "holdout_segments": len(held),
        "identities": len({s["group"] for s in train}),
        "clips": clips,
        "storage_gb": round((clips * CLIP_MB_512_49 * scale
                             + (sources.total_hours(train) + sources.total_hours(held)) * 3600 * CROP_MB_PER_S)
                            / 1024, 1),
    }
    plan = {"stage": asdict(cfg), "estimate": est, "train": train, "holdout": held}
    with open(paths.p("plan.json"), "w") as f:
        json.dump(plan, f)
    if est["storage_gb"] > cfg.storage_budget_gb and not force:
        raise RuntimeError(f"stage {cfg.name!r} needs ~{est['storage_gb']} GB, budget is {cfg.storage_budget_gb} GB "
                           "(raise storage_budget_gb, use a bigger disk, or pass force=True)")
    return plan


def n_clips(seg: dict, c: ClipConfig) -> int:
    minutes = (seg["end"] - seg["start"]) / 60
    return int(max(1, min(c.max_clips_per_video, round(minutes * c.clips_per_minute))))


# ----------------------------------------------------------------------------- 2. acquire
def acquire(plan: dict, work: str, cookies: str | None = None, downloader=None, cropper=None, detector=None,
            limit: int | None = None, retry_failed: bool = False, include_holdout: bool = True,
            tmp_dir: str | None = None, allow_youtube: bool = True, cleanup_segments: bool = False,
            progress=print) -> dict:
    """Download + face-crop every planned segment that is not on disk yet.

    Each segment comes from ``<work>/raw/segments/<segment id>.mp4`` (downloaded on your own computer with
    ``scripts/download_segments.py`` and uploaded / synced to Drive), else is cut from a full video in
    ``<work>/raw/sources/<youtube id>.mp4``, else is fetched with yt-dlp (``cookies`` = a cookies.txt export helps
    against YouTube's bot check). ``cleanup_segments`` deletes a downloaded segment once its face crop exists."""
    from .download import make_downloader
    from .face_crop import face_crop_video

    segments_dir = os.path.join(work, "raw", "segments")
    downloader = downloader or make_downloader(os.path.join(work, "raw", "sources"), allow_youtube, segments_dir)
    cropper = cropper or face_crop_video
    paths = StagePaths(work, plan["stage"]["name"])
    log_path = os.path.join(paths.raw, "acquire_log.jsonl")
    failed = set()  # only permanent failures (no usable face) are skipped; download errors are retried
    if os.path.isfile(log_path) and not retry_failed:
        for line in open(log_path):
            if line.strip():
                e = json.loads(line)
                if e.get("error") == "no stable face":
                    failed.add(e["id"])
    jobs = [(s, False) for s in plan["train"]] + ([(s, True) for s in plan["holdout"]] if include_holdout else [])
    counts = {"ok": 0, "skipped": 0, "failed": 0, "existing": 0}
    tmp_root = tmp_dir or tempfile.mkdtemp(prefix="wilrosync_dl_")
    os.makedirs(tmp_root, exist_ok=True)
    done = 0
    for seg, is_held in jobs:
        out = raw_path(paths, seg, is_held)
        if os.path.isfile(out):
            counts["existing"] += 1
            continue
        if seg["id"] in failed:
            counts["skipped"] += 1
            continue
        if limit is not None and done >= limit:
            break
        done += 1
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = os.path.join(tmp_root, seg["id"] + ".mp4")
        entry = {"id": seg["id"]}
        try:
            downloader(seg["url"], seg["start"], seg["end"], tmp, cookies=cookies, seg_id=seg["id"])
            if detector is None and cropper is face_crop_video:
                from ..eval.syncnet import S3FDDetector

                detector = S3FDDetector()
            info = cropper(tmp, out + ".part.mp4", detector=detector, size=plan["stage"]["clips"]["size"])
            if info is None:
                entry["error"] = "no stable face"
            else:
                os.replace(out + ".part.mp4", out)
                entry.update(info)
                seg_file = os.path.join(segments_dir, seg["id"] + ".mp4")
                if cleanup_segments and os.path.isfile(seg_file):  # the face crop replaces it
                    os.remove(seg_file)
        except Exception as e:  # keep going
            entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            for f in (tmp, out + ".part.mp4"):
                if os.path.exists(f):
                    os.remove(f)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        counts["failed" if "error" in entry else "ok"] += 1
        if progress:
            progress(f"[{done}] {seg['id']}: {entry.get('error', 'ok')}")
    if plan["stage"]["mead"]["enabled"]:
        counts["mead"] = crop_mead(plan, work, cropper=cropper, detector=detector, progress=progress)
    if progress:
        progress(f"acquire: {counts}")
    return counts


def _mead_pairs(plan: dict, paths: StagePaths) -> list[dict]:
    m = plan["stage"]["mead"]
    root = os.path.join(paths.raw, m["root"])
    if not os.path.isdir(root):
        return []
    return sources.mead_pairs(root, tuple(m["views"]), m["per_folder"], plan["stage"]["seed"], relative_to=paths.raw)


def mead_crop_path(paths: StagePaths, rel: str) -> str:
    return os.path.join(paths.raw, "mead_crop", rel)


def crop_mead(plan: dict, work: str, cropper=None, detector=None, progress=print) -> dict:
    """Face-crop every MEAD clip used by this stage's pseudo pairs (local files, no download)."""
    from .face_crop import face_crop_video

    cropper = cropper or face_crop_video
    paths = StagePaths(work, plan["stage"]["name"])
    pairs = _mead_pairs(plan, paths)
    if not pairs:
        if progress:
            progress(f"MEAD enabled but nothing under {os.path.join(paths.raw, plan['stage']['mead']['root'])}; "
                     "download it manually (docs/DATASETS.md)")
        return {"pairs": 0}
    files = sorted({r[k] for r in pairs for k in ("cond", "target")})
    counts = {"pairs": len(pairs), "ok": 0, "existing": 0, "failed": 0}
    for rel in files:
        out = mead_crop_path(paths, rel)
        if os.path.isfile(out):
            counts["existing"] += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            if detector is None and cropper is face_crop_video:
                from ..eval.syncnet import S3FDDetector

                detector = S3FDDetector()
            info = cropper(os.path.join(paths.raw, rel), out + ".part.mp4", detector=detector,
                           size=plan["stage"]["clips"]["size"])
            if info is None:
                counts["failed"] += 1
            else:
                os.replace(out + ".part.mp4", out)
                counts["ok"] += 1
        except Exception as e:
            counts["failed"] += 1
            if progress:
                progress(f"[mead] {rel}: {e}")
        finally:
            if os.path.exists(out + ".part.mp4"):
                os.remove(out + ".part.mp4")
    return counts


def raw_path(paths: StagePaths, seg: dict, holdout: bool = False) -> str:
    sub = os.path.join("holdout", seg["source"]) if holdout else seg["source"]
    return os.path.join(paths.raw, sub, seg["id"] + ".mp4")


# ----------------------------------------------------------------------------- 3. manifests
def write_manifests(plan: dict, work: str) -> dict:
    """Training manifest of segments that made it to disk, MEAD pairs, and the held-out list."""
    cfg = plan["stage"]
    paths = StagePaths(work, cfg["name"])
    c = ClipConfig(**cfg["clips"])
    wild = []
    for seg in plan["train"]:
        p = raw_path(paths, seg)
        if os.path.isfile(p):
            wild.append({"kind": "arbitrary", "id": seg["id"], "video": os.path.relpath(p, paths.raw),
                         "caption": c.caption, "clips": n_clips(seg, c), "group": seg["group"],
                         "mouth": CROP_MOUTH, "face_w": CROP_FACE_W})
    for folder in cfg.get("extra_videos") or []:  # your own (already face-cropped) videos
        exts = (".mp4", ".mov", ".mkv", ".webm", ".avi")
        for root, _, files in os.walk(os.path.join(paths.raw, folder)):
            for fn in sorted(files):
                if fn.lower().endswith(exts):
                    fp = os.path.join(root, fn)
                    wild.append({"kind": "arbitrary", "id": "extra_" + os.path.relpath(fp, paths.raw),
                                 "video": os.path.relpath(fp, paths.raw), "caption": c.caption,
                                 "clips": c.max_clips_per_video // 2 or 1, "group": "extra"})
    held = [{"id": s["id"], "video": os.path.relpath(raw_path(paths, s, True), paths.raw), "group": s["group"]}
            for s in plan["holdout"] if os.path.isfile(raw_path(paths, s, True))]
    pairs = []
    if cfg["mead"]["enabled"]:
        for r in _mead_pairs(plan, paths):
            cond, target = mead_crop_path(paths, r["cond"]), mead_crop_path(paths, r["target"])
            if os.path.isfile(cond) and os.path.isfile(target):  # face-cropped by acquire()
                pairs.append({"kind": "pseudo_pair", "cond": os.path.relpath(cond, paths.raw),
                              "target": os.path.relpath(target, paths.raw), "caption": c.caption,
                              "group": r["group"], "mouth": CROP_MOUTH, "face_w": CROP_FACE_W})
    out = {"wild": paths.p("wild.jsonl"), "pairs": paths.p("pairs.jsonl"), "holdout": paths.p("holdout.jsonl")}
    for key, rows in (("wild", wild), ("pairs", pairs), ("holdout", held)):
        with open(out[key], "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    out["counts"] = {"wild": len(wild), "pairs": len(pairs), "holdout": len(held),
                     "wild_hours": round(sources.total_hours([s for s in plan["train"]
                                                              if os.path.isfile(raw_path(paths, s))]), 2)}
    return out


# ----------------------------------------------------------------------------- 4 + 5. sync filter, prepare
def sync_and_prepare(plan: dict, work: str, manifests: dict, scorer=None, encoders=None, device: str = "cuda",
                     prepare: bool = True) -> dict:
    from .prepare import main as prepare_main
    from .sync_filter import filter_manifest

    cfg = plan["stage"]
    paths = StagePaths(work, cfg["name"])
    result = {}
    for key in ("wild", "pairs"):
        man = manifests[key]
        if not os.path.isfile(man) or os.path.getsize(man) == 0:
            continue
        synced = man.replace(".jsonl", ".synced.jsonl")
        # one shared score cache for all stages
        report = os.path.join(work, "raw", f"syncnet_{key}.report.jsonl")
        result[f"{key}_sync"] = filter_manifest(man, paths.raw, synced, report=report,
                                                min_conf=cfg["sync"]["min_conf"],
                                                max_offset=cfg["sync"]["max_offset"],
                                                assume_cropped=True, scorer=scorer)  # all face crops
        if prepare and os.path.getsize(synced) > 0:
            c = cfg["clips"]
            index = paths.p(f"index_{key}.jsonl")
            result[f"{key}_clips"] = prepare_main(
                ["--manifest", synced, "--root", paths.raw, "--out", paths.clips, "--index", index,
                 "--size", str(c["size"]), str(c["size"]), "--frames", str(c["frames"]), "--device", device],
                encoders=encoders)
            result[f"{key}_index"] = index
    return result


def train_overrides(plan: dict, work: str) -> dict:
    """Overrides for ``load_config`` pointing training at this stage's data."""
    paths = StagePaths(work, plan["stage"]["name"])
    pairs = paths.p("index_pairs.jsonl")
    has_pairs = os.path.isfile(pairs) and os.path.getsize(pairs) > 0
    return {
        "data.arbitrary_index": paths.p("index_wild.jsonl"),
        "data.pairs_index": pairs if has_pairs else "null",
        "data.tds_enabled": has_pairs,
        "data.null_text": os.path.join(work, "latents", "null_text.safetensors"),
        "output_dir": os.path.join(work, "runs", plan["stage"]["name"]),
        **plan["stage"].get("train_overrides", {}),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="wilro-sync data", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", help="smoke | first_model | paper_scale | path/to/stage.yaml")
    p.add_argument("--work", required=True, help="working folder (raw videos, clips, stages)")
    p.add_argument("--steps", default="plan,acquire,manifest,prepare",
                   help="comma list of plan, acquire, manifest, prepare")
    p.add_argument("--cookies", default=None, help="YouTube cookies.txt for yt-dlp")
    p.add_argument("--limit", type=int, default=None, help="acquire at most N new segments this run")
    p.add_argument("--no-youtube", action="store_true", help="only cut from <work>/raw/sources (no yt-dlp)")
    p.add_argument("--export-urls", default=None, help="write the YouTube URLs still needed to this file and exit")
    p.add_argument("--force", action="store_true", help="ignore the storage budget")
    p.add_argument("--device", default="cuda")
    p.add_argument("overrides", nargs="*", help="stage overrides, e.g. talkvid.hours=50")
    a = p.parse_args(argv)
    steps = set(a.steps.split(","))
    cfg = load_stage(a.stage, a.overrides)
    plan_file = os.path.join(a.work, "stages", cfg.name, "plan.json")
    if "plan" in steps or not os.path.isfile(plan_file):
        plan = plan_stage(cfg, a.work, force=a.force)
        print(f"plan {cfg.name}: {plan['estimate']}")
    else:
        plan = json.load(open(plan_file))
    if a.export_urls:
        from .download import export_url_list

        n = export_url_list(plan["train"] + plan["holdout"], a.export_urls, os.path.join(a.work, "raw", "sources"))
        print(f"{n} videos still needed -> {a.export_urls}")
        return
    if "acquire" in steps:
        acquire(plan, a.work, cookies=a.cookies, limit=a.limit, allow_youtube=not a.no_youtube)
    manifests = write_manifests(plan, a.work)
    print("manifests:", manifests["counts"])
    if "prepare" in steps:
        print(sync_and_prepare(plan, a.work, manifests, device=a.device))
    print("train with:", " ".join(f"{k}={v}" for k, v in train_overrides(plan, a.work).items()))


if __name__ == "__main__":
    main(sys.argv[1:])
