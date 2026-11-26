#!/usr/bin/env python3
"""Client for the wilro-sync HTTP server (stdlib only, no requests needed).

Sync (hold the connection):

    python scripts/server_client.py http://192.168.1.103:8000 video.mp4 audio.wav \
        --bbox 575,140,675,255 -o out.mp4 --key MYKEY

Background job (the robust way behind proxies / on Colab):

    python scripts/server_client.py http://192.168.1.103:8000 video.mp4 audio.wav \
        --bbox 575,140,675,255 --bg -o out.mp4 --key MYKEY

Check the crop first (no model, seconds):

    python scripts/server_client.py http://192.168.1.103:8000 video.mp4 audio.wav \
        --preview -o mask.mp4 --key MYKEY

Server status (no key needed for /health):

    python scripts/server_client.py http://192.168.1.103:8000 --health

Every other pipeline setting goes through FIELD=VALUE pairs, exactly the fields in
GET / (steps=20, ckpt=step_0060500, ...). URL and key also come from the
WILROSYNC_URL / WILROSYNC_KEY environment variables.
"""

from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

USER_AGENT = "wilro-sync-client/0.1"


class ServerError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def multipart(fields: list[tuple[str, str | None, bytes, str | None]]) -> tuple[str, bytes]:
    """Encode (name, filename, data, content_type) parts; filename None -> plain field."""
    boundary = "----wilrosync" + uuid.uuid4().hex
    body = bytearray()
    for name, filename, data, ctype in fields:
        body += f"--{boundary}\r\n".encode()
        if filename is None:
            body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        else:
            ctype = ctype or mimetypes.guess_type(filename)[0] or "application/octet-stream"
            body += (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n"
            ).encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return boundary, bytes(body)


def call(
    url: str, key: str | None, fields: list[tuple[str, str | None, bytes, str | None]], timeout: int
) -> tuple[int, bytes]:
    boundary, body = multipart(fields)
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": USER_AGENT}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get(url: str, key: str | None, timeout: int = 30) -> tuple[int, bytes]:
    headers = {"User-Agent": USER_AGENT}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def detail(status: int, body: bytes) -> str:
    try:
        parsed = json.loads(body)
        return parsed.get("detail", body.decode("utf-8", "replace"))
    except (ValueError, AttributeError):
        return body.decode("utf-8", "replace")[:400]


NET_ERRORS = (OSError, http.client.HTTPException)  # URLError/RemoteDisconnected/reset/timeout


def poll_job(base: str, key: str | None, job_id: str, interval: float, deadline: float | None,
             max_streak: int = 10) -> dict:
    """Poll one job, riding out transient connection drops (server restart, flaky network)."""
    url = f"{base}/jobs/{job_id}"
    misses = 0
    while True:
        try:
            status, body = get(url, key)
        except NET_ERRORS as e:
            misses += 1
            if misses > max_streak:
                raise ServerError(503, f"lost the server while polling {job_id}: {e}") from e
            print(f"  network hiccup {misses}/{max_streak}: {e}", file=sys.stderr)
            time.sleep(max(interval, 5.0))
            continue
        misses = 0
        if status != 200:
            raise ServerError(status, detail(status, body))
        state = json.loads(body)
        if state.get("status") in ("done", "error"):
            return state
        print(f"  {state.get('status')} (queue_position={state.get('queue_position', '-')})",
              file=sys.stderr)
        if deadline is not None and time.monotonic() > deadline:
            raise ServerError(408, f"client gave up waiting for job {job_id}")
        time.sleep(interval)


def fetch_result(base: str, key: str | None, job_id: str, out: Path, timeout: int,
                 attempts: int = 3) -> None:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            status, body = get(f"{base}/jobs/{job_id}/result", key, timeout=timeout)
        except NET_ERRORS as e:  # a 410 on retry means the blob left with the failed transfer
            last = e
            print(f"  download hiccup {attempt}/{attempts}: {e}", file=sys.stderr)
            time.sleep(5)
            continue
        if status != 200:
            raise ServerError(status, detail(status, body))
        out.write_bytes(body)
        print(f"saved {out} ({len(body)} bytes)")
        return
    raise ServerError(503, f"could not download result of {job_id}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("server", nargs="?", default=os.environ.get("WILROSYNC_URL"),
                    help="e.g. http://192.168.1.103:8000 (or WILROSYNC_URL)")
    ap.add_argument("video", nargs="?", help="input video (mp4/...)")
    ap.add_argument("audio", nargs="?", help="audio track (wav/mp3); required unless --preview … the "
                                            "server still wants a short file for its length")
    ap.add_argument("--bbox", help="x0,y0,x1,y1 in first-frame pixels (the face to sync)")
    ap.add_argument("--preview", action="store_true", help="crop+mask only, no model (seconds)")
    ap.add_argument("--bg", action="store_true", help="POST /jobs and poll instead of a blocking call")
    ap.add_argument("--timeout", type=int, default=0, help="blocking call: seconds to wait before the "
                                                           "server returns 202 (0 = wait forever)")
    ap.add_argument("--poll-interval", type=float, default=3.0)
    ap.add_argument("--give-up", type=float, default=14400, help="client-side wait limit in seconds "
                                                                 "(default 4h: long GPU jobs)")
    ap.add_argument("-o", "--out", default="out.mp4")
    ap.add_argument("--key", default=os.environ.get("WILROSYNC_KEY"), help="X-API-Key (or WILROSYNC_KEY)")
    ap.add_argument("--health", action="store_true", help="just show /health and exit")
    ap.add_argument("fields", nargs="*", help="extra settings as FIELD=VALUE, e.g. steps=20")
    args = ap.parse_args()

    if not args.server:
        ap.error("server URL is required (or set WILROSYNC_URL)")
    if not args.server.startswith(("http://", "https://")):
        # first positional was the video (WILROSYNC_URL style): shift the positionals left
        if not os.environ.get("WILROSYNC_URL"):
            ap.error("first argument is not a URL and WILROSYNC_URL is not set")
        args.video, args.audio, extra = args.server, args.video, args.audio
        args.server = os.environ["WILROSYNC_URL"]
        if extra:
            args.fields.insert(0, extra)
    base = args.server.rstrip("/")
    if args.health:
        status, body = get(f"{base}/health", None)
        print(json.dumps(json.loads(body), indent=2))
        return 0 if status == 200 else 1

    if not args.video:
        ap.error("video is required (or use --health)")

    fields: list[tuple[str, str | None, bytes, str | None]] = []
    video_path = Path(args.video)
    fields.append(("video", video_path.name, video_path.read_bytes(), None))
    if args.audio:
        audio_path = Path(args.audio)
        fields.append(("audio", audio_path.name, audio_path.read_bytes(), None))
    for name, value in (("--bbox", args.bbox), ("--preview", "true" if args.preview else None),
                        ("--timeout", str(args.timeout) if args.timeout else None)):
        if value is not None:
            fields.append((name.lstrip("-"), None, value.encode(), None))
    for spec in args.fields:
        name, sep, value = spec.partition("=")
        if not sep:
            ap.error(f"extra fields look like FIELD=VALUE, got {spec!r}")
        fields.append((name, None, value.encode(), None))

    def run_bg(endpoint: str) -> int:
        try:
            status, body = call(f"{base}{endpoint}", args.key, fields, timeout=120)
        except NET_ERRORS as e:
            # never auto-retry a submit: the job may exist server-side; check /jobs first
            raise ServerError(503, f"could not reach the server to submit: {e} "
                                   f"(check GET {base}/jobs before resubmitting)") from e
        if status not in (200, 202):
            raise ServerError(status, detail(status, body))
        job = json.loads(body)
        print(f"job {job['job_id']} (queue_position={job.get('queue_position')})", file=sys.stderr)
        state = poll_job(base, args.key, job["job_id"], args.poll_interval,
                         time.monotonic() + args.give_up)
        if state["status"] != "done":
            print(f"job failed: {state.get('error')}", file=sys.stderr)
            return 1
        if state.get("inference_seconds"):
            print(f"inference: {state['inference_seconds']:.1f}s", file=sys.stderr)
        fetch_result(base, args.key, job["job_id"], Path(args.out), timeout=600)
        return 0

    if args.bg:  # background job, robust behind proxies / on Colab
        return run_bg("/jobs")
    if args.timeout > 0:  # blocking, but the server hands it over after --timeout seconds
        status, body = call(f"{base}/lipsync", args.key, fields, timeout=120 + args.timeout)
        if status == 202:
            job_id = json.loads(body)["job_id"]
            print(f"job {job_id} still running after {args.timeout}s; polling", file=sys.stderr)
            state = poll_job(base, args.key, job_id, args.poll_interval,
                             time.monotonic() + args.give_up)
            if state["status"] != "done":
                print(f"job failed: {state.get('error')}", file=sys.stderr)
                return 1
            fetch_result(base, args.key, job_id, Path(args.out), timeout=600)
            return 0
        if status == 200:
            Path(args.out).write_bytes(body)
            print(f"saved {args.out} ({len(body)} bytes)")
            return 0
        raise ServerError(status, detail(status, body))
    # plain blocking call: hold the connection until the video is ready
    status, body = call(f"{base}/lipsync", args.key, fields, timeout=120 + args.give_up)
    if status == 200:
        Path(args.out).write_bytes(body)
        print(f"saved {args.out} ({len(body)} bytes)")
        return 0
    if status == 202:  # server-side cap kicked in anyway
        job_id = json.loads(body)["job_id"]
        state = poll_job(base, args.key, job_id, args.poll_interval,
                         time.monotonic() + args.give_up)
        if state["status"] != "done":
            print(f"job failed: {state.get('error')}", file=sys.stderr)
            return 1
        fetch_result(base, args.key, job_id, Path(args.out), timeout=600)
        return 0
    raise ServerError(status, detail(status, body))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ServerError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
