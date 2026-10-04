"""SyncNet filtering of a clip manifest (milestone M3).

Every video in the manifest is scored with SyncNet. Rows are annotated with

    av_offset  A/V offset in frames at 25 fps (prepare_clips shifts the audio windows by -av_offset)
    sync_conf  SyncNet confidence (LSE-C)
    sync_dist  SyncNet distance at the best offset (LSE-D)

and only rows with ``sync_conf >= min_conf`` and ``|av_offset| <= max_offset`` are written to the filtered
manifest. A full report (including rejected rows and errors) is written alongside and reused on re-runs, so an
interrupted job (e.g. a Colab disconnect) resumes where it stopped.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable


def _video_of(row: dict) -> str:
    # pseudo pairs: the target clip carries the audio that drives the lips
    return row["target"] if row.get("kind") == "pseudo_pair" else row["video"]


def filter_manifest(
    manifest: str,
    root: str,
    out_manifest: str,
    report: str | None = None,
    min_conf: float = 3.0,
    max_offset: int = 3,
    max_seconds: float | None = 30.0,
    assume_cropped: bool = False,
    det_every: int = 2,
    scorer=None,
    progress: Callable[[str], None] | None = print,
) -> dict:
    """Score, annotate and filter ``manifest`` -> ``out_manifest``. Returns counts."""
    report = report or os.path.splitext(out_manifest)[0] + ".report.jsonl"
    done: dict[str, dict] = {}
    if os.path.isfile(report):
        for line in open(report):
            if line.strip():
                r = json.loads(line)
                done[r["_scored"]] = r
    rows = [json.loads(line) for line in open(manifest) if line.strip()]
    todo = [r for r in rows if _video_of(r) not in done]
    if todo and scorer is None:
        from ..eval.syncnet import SyncNet

        scorer = SyncNet()
    os.makedirs(os.path.dirname(os.path.abspath(out_manifest)) or ".", exist_ok=True)
    with open(report, "a") as rep:
        for i, row in enumerate(todo):
            rel = _video_of(row)
            res = dict(row, _scored=rel)
            try:
                r = scorer.score_file(os.path.join(root, rel), max_seconds=max_seconds,
                                      assume_cropped=assume_cropped, det_every=det_every)
                if r is None:
                    res["error"] = "no face track"
                else:
                    res.update(av_offset=int(r.offset), sync_conf=round(float(r.conf), 4),
                               sync_dist=round(float(r.min_dist), 4), sync_frames=int(r.num_frames))
            except Exception as e:  # keep going on bad files
                res["error"] = f"{type(e).__name__}: {e}"
            rep.write(json.dumps(res) + "\n")
            rep.flush()
            done[rel] = res
            if progress:
                msg = res.get("error") or f"offset {res['av_offset']:+d}  conf {res['sync_conf']:.2f}"
                progress(f"[{i + 1}/{len(todo)}] {rel}: {msg}")

    kept = 0
    counts = {"total": len(rows), "kept": 0, "low_conf": 0, "offset": 0, "error": 0}
    with open(out_manifest, "w") as out:
        for row in rows:
            r = done[_video_of(row)]
            if "error" in r:
                counts["error"] += 1
            elif r["sync_conf"] < min_conf:
                counts["low_conf"] += 1
            elif abs(r["av_offset"]) > max_offset:
                counts["offset"] += 1
            else:
                keep = dict(row, av_offset=r["av_offset"], sync_conf=r["sync_conf"], sync_dist=r["sync_dist"])
                out.write(json.dumps(keep) + "\n")
                kept += 1
    counts["kept"] = kept
    if progress:
        progress(f"kept {kept}/{len(rows)} (low confidence {counts['low_conf']}, offset > {max_offset}: "
                 f"{counts['offset']}, errors {counts['error']}) -> {out_manifest}")
    return counts


def main(argv: list[str] | None = None) -> dict:
    import argparse

    p = argparse.ArgumentParser(prog="wilro-sync sync-filter", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", default=".", help="video paths in the manifest are relative to this folder")
    p.add_argument("--out", required=True, help="filtered manifest (.jsonl)")
    p.add_argument("--report", default=None, help="all scores incl. rejected rows (default: <out>.report.jsonl)")
    p.add_argument("--min-conf", type=float, default=3.0)
    p.add_argument("--max-offset", type=int, default=3)
    p.add_argument("--max-seconds", type=float, default=30.0, help="score only the first N seconds (0 = all)")
    p.add_argument("--assume-cropped", action="store_true", help="videos are already face crops (skip detection)")
    p.add_argument("--det-every", type=int, default=2, help="run the face detector every N frames")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    from ..eval.syncnet import SyncNet

    return filter_manifest(a.manifest, a.root, a.out, a.report, a.min_conf, a.max_offset, a.max_seconds or None,
                           a.assume_cropped, a.det_every, scorer=SyncNet(device=a.device))


if __name__ == "__main__":
    main(sys.argv[1:])
