"""Download one time range of a YouTube video with yt-dlp (``pip install yt-dlp``).

Respect YouTube's terms and the dataset licences. On cloud machines (e.g. Colab) YouTube often asks to
"confirm you're not a bot": export your browser cookies to a Netscape cookies.txt and pass ``cookies=``.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from urllib.parse import parse_qs, urlparse


def yt_dlp_available() -> bool:
    if shutil.which("yt-dlp"):
        return True
    try:
        import yt_dlp  # noqa: F401

        return True
    except ImportError:
        return False


def download_segment(url: str, start: float, end: float, out_path: str, cookies: str | None = None,
                     max_height: int = 1080, accurate_cuts: bool = False, timeout: int = 900) -> None:
    """Download ``url`` between ``start`` and ``end`` seconds to ``out_path`` (mp4). Raises on failure."""
    from ..io.media import _ffmpeg

    exe = shutil.which("yt-dlp")
    cmd = [exe] if exe else ["python", "-m", "yt_dlp"]
    tmpl = out_path + ".dl.%(ext)s"
    cmd += [
        "-q", "--no-warnings", "--no-playlist", "--no-part",
        "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start:.3f}-{end:.3f}",
        "--ffmpeg-location", _ffmpeg(),
        "-o", tmpl,
    ]
    if accurate_cuts:
        cmd.append("--force-keyframes-at-cuts")
    if cookies:
        cmd += ["--cookies", cookies]
    cmd.append(url)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    got = out_path + ".dl.mp4"
    if proc.returncode or not os.path.isfile(got):
        for ext in ("mp4", "webm", "mkv"):
            if os.path.exists(out_path + f".dl.{ext}"):
                os.remove(out_path + f".dl.{ext}")
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(err[-1] if err else f"yt-dlp failed ({proc.returncode})")
    os.replace(got, out_path)


def youtube_id(url: str) -> str | None:
    u = urlparse(url)
    if u.hostname and "youtu.be" in u.hostname:
        return u.path.lstrip("/") or None
    q = parse_qs(u.query).get("v")
    if q:
        return q[0]
    m = re.search(r"/(?:shorts|embed)/([\w-]{11})", u.path)
    return m.group(1) if m else None


def find_local_source(url: str, sources_dir: str | None) -> str | None:
    """A full video downloaded beforehand as ``<sources_dir>/<youtube id>.<ext>``."""
    vid = youtube_id(url)
    if not sources_dir or not vid:
        return None
    hits = [p for p in glob.glob(os.path.join(sources_dir, glob.escape(vid) + ".*")) if not p.endswith(".part")]
    return hits[0] if hits else None


def cut_segment(src: str, start: float, end: float, out_path: str) -> None:
    """Cut [start, end] seconds out of a local video (re-encoded so the cut is frame accurate)."""
    from ..io.media import ffmpeg

    ffmpeg("-ss", f"{start:.3f}", "-i", src, "-t", f"{end - start:.3f}", "-c:v", "libx264", "-crf", "16",
           "-preset", "veryfast", "-c:a", "aac", out_path)


def make_downloader(sources_dir: str | None = None, allow_youtube: bool = True):
    """Downloader that cuts from a local full video when available, otherwise uses yt-dlp."""

    def get(url: str, start: float, end: float, out_path: str, cookies: str | None = None) -> None:
        local = find_local_source(url, sources_dir)
        if local:
            cut_segment(local, start, end, out_path)
        elif allow_youtube:
            download_segment(url, start, end, out_path, cookies=cookies)
        else:
            raise FileNotFoundError(f"no local source for {url} in {sources_dir}")

    return get


def export_url_list(segments: list[dict], path: str, sources_dir: str | None = None) -> int:
    """Write the unique YouTube URLs still needed (one per line), e.g. for
    ``yt-dlp -a urls.txt -f "bv*[height<=1080]+ba/b" --merge-output-format mp4 -o "%(id)s.%(ext)s"``
    on your own computer, then upload the files into ``sources_dir``."""
    urls = []
    seen = set()
    for s in segments:
        vid = youtube_id(s["url"])
        if vid and vid not in seen and not find_local_source(s["url"], sources_dir):
            seen.add(vid)
            urls.append(f"https://www.youtube.com/watch?v={vid}")
    with open(path, "w") as f:
        f.writelines(u + "\n" for u in urls)
    return len(urls)
