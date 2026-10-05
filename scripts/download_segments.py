#!/usr/bin/env python3
"""Download a data stage's video segments on your own computer (e.g. a Mac at home, where YouTube does not
block downloads), straight into a Google Drive folder.

Only needs Python 3.8+, yt-dlp and ffmpeg (macOS: ``brew install yt-dlp ffmpeg``). No wilro-sync install.

1. In Colab, run the "plan" cell of the notebook. It writes
   ``<WORK_DIR>/stages/<stage>/plan.json`` to your Drive.
2. On your computer (with Google Drive for desktop, the Drive folder is a normal local folder):

       python3 download_segments.py \\
           --plan  "~/Google Drive/My Drive/wilro-sync-work/stages/smoke/plan.json" \\
           --out   "~/Google Drive/My Drive/wilro-sync-work/raw/segments"

   Without Drive for desktop, use any local --out folder and upload it to ``<WORK_DIR>/raw/segments`` in the
   Drive web UI.
3. Back in Colab, run the "acquire" cell: segments found in ``raw/segments`` are used instead of YouTube.

Re-running skips segments that already exist; failures are listed in ``<out>/failed.txt``. Respect YouTube's
terms and the dataset licences.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def yt_dlp_cmd() -> list:
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401

        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        sys.exit("yt-dlp not found: brew install yt-dlp  (or: pip install yt-dlp)")


def download(seg: dict, out_dir: str, args) -> tuple:
    out = os.path.join(out_dir, seg["id"] + ".mp4")
    if os.path.isfile(out):
        return seg["id"], "exists"
    tmpl = os.path.join(out_dir, ".tmp_" + seg["id"] + ".%(ext)s")
    cmd = yt_dlp_cmd() + [
        "-q", "--no-warnings", "--no-playlist", "--no-part",
        "-f", f"bv*[height<={args.max_height}]+ba/b[height<={args.max_height}]/b",
        "--merge-output-format", "mp4",
        "--download-sections", "*{:.3f}-{:.3f}".format(seg["start"], seg["end"]),
        "-o", tmpl,
    ]
    if args.cookies:
        cmd += ["--cookies", os.path.expanduser(args.cookies)]
    if args.cookies_from_browser:
        cmd += ["--cookies-from-browser", args.cookies_from_browser]
    if args.ffmpeg:
        cmd += ["--ffmpeg-location", args.ffmpeg]
    cmd.append(seg["url"])
    for attempt in range(args.retries + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        got = os.path.join(out_dir, ".tmp_" + seg["id"] + ".mp4")
        if proc.returncode == 0 and os.path.isfile(got):
            os.replace(got, out)
            return seg["id"], "ok"
        time.sleep(2 * (attempt + 1))
    for f in os.listdir(out_dir):
        if f.startswith(".tmp_" + seg["id"] + "."):
            os.remove(os.path.join(out_dir, f))
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return seg["id"], "error: " + (err[-1] if err else f"yt-dlp exit {proc.returncode}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True, help="stages/<stage>/plan.json written by the plan step")
    p.add_argument("--out", required=True, help="output folder, i.e. <WORK_DIR>/raw/segments")
    p.add_argument("--which", default="all", choices=["all", "train", "holdout"])
    p.add_argument("--limit", type=int, default=None, help="download at most N segments this run")
    p.add_argument("--workers", type=int, default=2, help="parallel downloads (keep small to avoid rate limits)")
    p.add_argument("--max-height", type=int, default=1080)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--cookies", default=None, help="cookies.txt (Netscape format)")
    p.add_argument("--cookies-from-browser", default=None, help="e.g. chrome, safari, firefox")
    p.add_argument("--ffmpeg", default=None, help="path to ffmpeg if it is not on PATH")
    args = p.parse_args()

    if not shutil.which("ffmpeg") and not args.ffmpeg:
        sys.exit("ffmpeg not found: brew install ffmpeg  (or pass --ffmpeg /path/to/ffmpeg)")
    with open(os.path.expanduser(args.plan)) as f:
        plan = json.load(f)
    segs = []
    if args.which in ("all", "train"):
        segs += plan["train"]
    if args.which in ("all", "holdout"):
        segs += plan["holdout"]
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    todo = [s for s in segs if not os.path.isfile(os.path.join(out_dir, s["id"] + ".mp4"))]
    if args.limit is not None:
        todo = todo[: args.limit]
    hours = sum(s["end"] - s["start"] for s in todo) / 3600
    print(f"{len(segs)} segments in plan, {len(todo)} to download (~{hours:.1f} h of video)")

    failed, done = [], 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(download, s, out_dir, args) for s in todo]
        for fut in as_completed(futures):
            sid, status = fut.result()
            done += 1
            print(f"[{done}/{len(todo)}] {sid}: {status}", flush=True)
            if status.startswith("error"):
                failed.append(f"{sid}\t{status}")
    with open(os.path.join(out_dir, "failed.txt"), "w") as f:
        f.write("\n".join(failed) + ("\n" if failed else ""))
    print(f"done: {len(todo) - len(failed)} ok, {len(failed)} failed (see failed.txt)")
    if any("not a bot" in x or "Sign in" in x for x in failed):
        print("YouTube wants a login: add --cookies-from-browser chrome (or safari / firefox)")


if __name__ == "__main__":
    main()
