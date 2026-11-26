"""HTTP service for face-crop lip-sync inference (``wilro-sync serve``).

The server loads one checkpoint and serves it over HTTP: send a video, an audio track and an
optional ``bbox`` (the face to sync, in first-frame pixels); get the lip-synced video back.
Settings default to the tuned Colab-notebook values (steps=10, window=129, overlap=2,
mode="lower", ...) and every field can be overridden per request.

Endpoints (all JSON errors are ``{"detail": ...}``):

    GET  /health            model status, device, queue depth (no API key needed)
    GET  /                  this summary + the request fields
    POST /lipsync           multipart form -> the lip-synced mp4 (or 202 + job_id when ``timeout`` is hit)
    POST /jobs              same form as a background job -> ``{"job_id": ...}``
    GET  /jobs/{id}         job status: queued | running | done | error
    GET  /jobs/{id}/result  the finished mp4 (kept for ``--keep-minutes``, default 60)
    GET  /jobs              all jobs (short form)

``POST /lipsync`` waits for the result, which is the simplest client; long jobs behind proxies
(Colab, ngrok) can outlive request timeouts, so the same request to ``POST /jobs`` returns at once
and the video is fetched from ``/jobs/{id}/result`` when ``status`` is ``"done"``.

Run:    wilro-sync serve --ckpt runs/first_model/step_0054800 --host 0.0.0.0 --port 8000
Deps:   pip install -e ".[server]"       (fastapi + uvicorn + python-multipart)
Docs:   docs/server.md
"""

from __future__ import annotations

import logging
import queue
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

logger = logging.getLogger("wilrosync.server")

# The tuned settings from the Colab notebook (cell 33 BASE); every one can be overridden per request.
SERVE_DEFAULTS: dict = {
    "steps": 10, "window": 129, "overlap": 2, "chain": 0, "tau": 0.92, "same_noise": False, "seed": 0,
    "omega_peak": 4.5, "gamma": 1.5,
    "mode": "lower", "loop": "reverse", "pause": 0, "crop": "auto", "crop_smooth": 9,
    "crop_deadzone": 0.15, "preroll": 4, "postroll": 8, "reference": "stride2", "blend": "double",
    "oval": "track", "track": {"width": 1.8, "bottom": 0.2, "top": 0.1, "spread": 0.2},
    "size": 512, "scale": 1.8,
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


class QueueFull(Exception):
    """More jobs are waiting than ``max_queue`` allows."""


# ------------------------------------------------------------------------------- request parsing
def parse_bbox(s: str) -> tuple[float, float, float, float]:
    """``"x0,y0,x1,y1"`` in first-frame pixels -> a tuple; raises ``ValueError``."""
    try:
        v = [float(x) for x in s.replace(" ", "").split(",")]
    except ValueError:
        raise ValueError(f"bbox must be 4 numbers x0,y0,x1,y1, got {s!r}") from None
    if len(v) != 4 or not all(x == x and abs(x) != float("inf") for x in v):
        raise ValueError(f"bbox must be 4 finite numbers, got {s!r}")
    if v[2] <= v[0] or v[3] <= v[1]:
        raise ValueError(f"bbox x1/y1 must be greater than x0/y0, got {s!r}")
    if min(v) < 0:
        raise ValueError(f"bbox must be non-negative, got {s!r}")
    return tuple(v)  # type: ignore[return-value]


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "on")


def _as_int(data: Mapping, key: str, lo: int, hi: int, default: int | None = None) -> int | None:
    v = data.get(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        n = int(str(v).strip())
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {v!r}") from None
    if not lo <= n <= hi:
        raise ValueError(f"{key} must be {lo}..{hi}, got {n}")
    return n


def _as_float(data: Mapping, key: str, lo: float, hi: float, default: float | None = None) -> float | None:
    v = data.get(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        x = float(str(v).strip())
    except ValueError:
        raise ValueError(f"{key} must be a number, got {v!r}") from None
    if not lo <= x <= hi:
        raise ValueError(f"{key} must be in [{lo}, {hi}], got {x}")
    return x


def _as_choice(data: Mapping, key: str, choices: tuple, default: str) -> str:
    v = data.get(key)
    if v is None or str(v).strip() == "":
        return default
    v = str(v).strip().lower()
    if v not in choices:
        raise ValueError(f"{key} must be one of {list(choices)}, got {v!r}")
    return v


def _as_json_dict(data: Mapping, key: str, allowed: set[str]) -> dict | None:
    v = data.get(key)
    if v is None or str(v).strip() == "":
        return None
    import json

    try:
        d = json.loads(v) if isinstance(v, str) else dict(v)
    except Exception:
        raise ValueError(f"{key} must be a JSON object, got {v!r}") from None
    if not isinstance(d, dict) or not d:
        raise ValueError(f"{key} must be a non-empty JSON object, got {v!r}")
    bad = set(d) - allowed
    if bad:
        raise ValueError(f"{key} has unknown keys {sorted(bad)}; allowed: {sorted(allowed)}")
    return d


def parse_params(data: Mapping) -> dict:
    """Validate the request fields (strings from a multipart form) -> ``{"bbox", "preview", "ckpt",
    "timeout", "video_name", "cfg", "face"}``; raises ``ValueError`` with a readable message."""
    from .pipeline.oval import FIXED_DEFAULT, FIXED_OLD

    unknown = set(data) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"unknown field(s) {sorted(unknown)}; see GET / for the request fields")

    out: dict = {}
    out["preview"] = _as_bool(data.get("preview", "false"))
    out["bbox"] = parse_bbox(str(data["bbox"])) if data.get("bbox") not in (None, "") else None
    ck = data.get("ckpt")
    out["ckpt"] = str(ck).strip() if ck not in (None, "") and str(ck).strip() else None
    out["timeout"] = _as_float(data, "timeout", 0.0, 24 * 3600.0)

    dscfg = {}
    for key in ("omega_peak", "gamma"):
        v = _as_float(data, key, 0.0, 20.0 if key == "omega_peak" else 10.0, SERVE_DEFAULTS[key])
        if v is not None:
            dscfg[key] = v
    cfg: dict = {"dscfg": dscfg} if dscfg else {}
    steps = _as_int(data, "steps", 1, 100, SERVE_DEFAULTS["steps"])
    if steps is not None:
        cfg["steps"] = steps
    for key in ("window", "overlap", "chain"):
        v = _as_int(data, key, 0, 4097, SERVE_DEFAULTS[key])
        if v is not None:
            cfg[key] = v
    if (cfg.get("window", SERVE_DEFAULTS["window"])) % 4 != 1:
        raise ValueError(f"window must be 1 + 4k frames (49, 81, 129, ...), got {cfg.get('window')}")
    ch = cfg.get("chain", SERVE_DEFAULTS["chain"])
    if ch and ch % 4 != 1:
        raise ValueError(f"chain must be 0 or 1 + 4k (1, 5, 9, ...), got {ch}")
    for key in ("tau",):
        v = _as_float(data, key, 0.01, 1.0, SERVE_DEFAULTS[key])
        if v is not None:
            cfg[key] = v
    if _as_bool(data.get("same_noise", "false")):
        cfg["same_noise"] = True
    seed = _as_int(data, "seed", -(2**62), 2**62, None)
    if seed is not None:
        cfg["seed"] = seed
    prompt = data.get("prompt")
    if prompt:
        cfg["prompt"] = str(prompt)[:500]
    out["cfg"] = cfg

    face: dict = {"mode": _as_choice(data, "mode", ("lower", "full"), SERVE_DEFAULTS["mode"]),
                  "loop": _as_choice(data, "loop", ("reverse", "restart", "hold", "trim"), SERVE_DEFAULTS["loop"]),
                  "crop": _as_choice(data, "crop", ("auto", "fixed", "track"), SERVE_DEFAULTS["crop"]),
                  "reference": _as_choice(data, "reference", ("same", "shift", "reverse", "stride2"),
                                          SERVE_DEFAULTS["reference"]),
                  "blend": _as_choice(data, "blend", ("double", "single"), SERVE_DEFAULTS["blend"])}
    face["pause"] = _as_int(data, "pause", 0, 300, SERVE_DEFAULTS["pause"])
    face["preroll"] = _as_int(data, "preroll", 0, 256, SERVE_DEFAULTS["preroll"])
    face["postroll"] = _as_int(data, "postroll", 0, 256, SERVE_DEFAULTS["postroll"])
    face["crop_smooth"] = _as_int(data, "crop_smooth", 0, 100, SERVE_DEFAULTS["crop_smooth"])
    face["crop_deadzone"] = _as_float(data, "crop_deadzone", 0.0, 2.0, SERVE_DEFAULTS["crop_deadzone"])
    det_every = _as_int(data, "det_every", 1, 100, None)
    if det_every is not None:
        face["det_every"] = det_every
    size = _as_int(data, "size", 64, 1024, SERVE_DEFAULTS["size"])
    if size % 8:
        raise ValueError(f"size must be a multiple of 8, got {size}")
    face["size"] = size
    face["scale"] = _as_float(data, "scale", 1.0, 4.0, SERVE_DEFAULTS["scale"])

    oval = str(data.get("oval") or SERVE_DEFAULTS["oval"]).strip().lower()
    if oval == "track":
        face["oval"] = "track"
    elif oval in ("fixed", "fixed_old"):
        face["oval"] = dict(FIXED_DEFAULT if oval == "fixed" else FIXED_OLD)
    elif oval.startswith("{"):
        d = _as_json_dict(data, "oval", {"cx", "cy", "rx", "ry", "feather", "angle"})
        face["oval"] = d
    else:
        raise ValueError(f"oval must be track | fixed | fixed_old | a JSON oval, got {oval!r}")
    track = _as_json_dict(data, "track", {"width", "bottom", "top", "feather", "spread", "spread_up", "smooth"})
    face["track"] = track if track is not None else dict(SERVE_DEFAULTS["track"])
    face["preview"] = out["preview"]
    out["face"] = face
    return out


ALLOWED_FIELDS = frozenset(
    ("video", "audio", "bbox", "preview", "ckpt", "timeout", "prompt", "steps", "tau", "window",
     "overlap", "chain", "same_noise", "seed", "omega_peak", "gamma", "mode", "loop", "pause", "crop",
     "crop_smooth", "crop_deadzone", "det_every", "preroll", "postroll", "reference", "blend", "oval",
     "track", "size", "scale"))


# ------------------------------------------------------------------------------- service
@dataclass
class ServerConfig:
    ckpt: str | None = None            # checkpoint dir (or a runs/ dir; the newest step_* is used)
    device: str | None = None          # None: cuda when available, else cpu
    dtype: str = "bf16"                # model dtype: bf16 | fp16 | fp32
    vae_bf16: bool = False
    text_device: str | None = None     # e.g. cpu to keep the 11 GB umT5 off the GPU
    text_embeds: str | None = None     # saved prompt embedding: never load umT5 (Jetson)
    api_key: str | None = None         # when set, requests need the X-API-Key header
    max_queue: int = 8                 # jobs waiting (the GPU runs one at a time)
    ckpt_root: str | None = None       # allow per-request `ckpt` inside this dir
    keep_minutes: float = 60.0         # keep finished jobs' results for this long
    max_upload_mb: int = 2048
    cors: bool = False                 # allow browser calls from any origin
    load_at_startup: bool = True


class Job:
    def __init__(self, jid: str, params: dict, tmpdir: Path):
        self.id, self.params, self.tmpdir = jid, params, tmpdir
        self.status = "queued"
        self.error: str | None = None
        self.client_error = False
        self.out: Path | None = None
        self.delivered = False  # the video has been sent once; /result then says 410 Gone
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.seconds: float | None = None
        self.event = threading.Event()


class InferenceService:
    """One checkpoint, one GPU job at a time, jobs kept in memory with their temp dirs."""

    def __init__(self, config: ServerConfig, infer_fn=None, load_pipe_fn=None):
        self.config = config
        self.infer_fn = infer_fn or _default_infer_fn
        self.load_pipe_fn = load_pipe_fn or _default_load_pipe
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pipe = None
        self._pipe_ckpt: str | None = None
        self._load_error: str | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- checkpoints
    def resolve_ckpt(self, override: str | None) -> str:
        base = override or self.config.ckpt
        if override and not self.config.ckpt_root:
            raise ValueError("this server does not allow per-request ckpt (start it with --ckpt-root)")
        if override and self.config.ckpt_root:
            root, p = Path(self.config.ckpt_root).resolve(), (Path(self.config.ckpt_root) / override).resolve()
            if not (p == root or root in p.parents):
                raise ValueError(f"ckpt {override!r} is outside --ckpt-root")
            base = str(p)
        if not base:
            raise ValueError("no checkpoint configured (start the server with --ckpt)")
        path = Path(base)
        if (path / "wilrosync_config.json").is_file():
            return str(path)
        steps = sorted(path.glob("step_*"))
        if steps:  # a runs/ dir: use its newest step_* (same as the notebook's latest_checkpoint)
            return str(steps[-1])
        raise ValueError(f"no checkpoint found at {base!r} (need a step_* dir or wilrosync_config.json)")

    def pipe(self, ckpt: str):
        if self._pipe is not None and self._pipe_ckpt == ckpt:
            return self._pipe
        with self._lock:
            if self._pipe is not None and self._pipe_ckpt == ckpt:
                return self._pipe
            try:
                pipe = self.load_pipe_fn(ckpt, self.config)
            except Exception as e:  # surfaced by /health and the job that triggered it
                self._load_error = f"{type(e).__name__}: {e}"
                raise
            self._load_error = None
            if self._pipe is not None:
                self._free()
            self._pipe, self._pipe_ckpt = pipe, ckpt
            return pipe

    def preload(self) -> None:
        try:
            ckpt = self.resolve_ckpt(None)
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            return
        try:
            self.pipe(ckpt)
        except Exception:
            logger.warning("checkpoint preload failed: %s", self._load_error)

    @staticmethod
    def _free() -> None:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    # -- jobs
    @property
    def model_loaded(self) -> bool:
        return self._pipe is not None

    def new_tmpdir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="wilrosync-"))

    def submit(self, params: dict, tmpdir: Path) -> Job:
        queued = sum(j.status == "queued" for j in self._jobs.values())
        if queued >= self.config.max_queue:
            raise QueueFull(f"queue is full ({queued} waiting); try again later")
        job = Job(uuid.uuid4().hex[:12], params, tmpdir)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def job(self, jid: str) -> Job | None:
        return self._jobs.get(jid)

    def jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created)

    def run_job(self, job: Job) -> None:  # runs in the worker thread
        from .pipeline.face import face_infer_config

        p = job.params
        job.status, job.started = "running", time.time()
        try:
            pipe = None if p["preview"] else self.pipe(self.resolve_ckpt(p["ckpt"]))
            started = time.time()
            out = self.infer_fn(pipe, p["video_path"], p["audio_path"], str(job.tmpdir / "out.mp4"),
                                bbox=p["bbox"], cfg=face_infer_config(**p["cfg"]), **p["face"])
            job.out, job.seconds = Path(out), time.time() - started
            job.status = "done"
            logger.info("job %s done in %.1fs -> %s", job.id, job.seconds, out)
        except Exception as e:  # noqa: BLE001 - any failure becomes the job's error
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
            job.client_error = isinstance(e, ValueError)
            logger.warning("job %s failed: %s", job.id, job.error)
        finally:
            job.finished = time.time()
            job.event.set()
            for key in ("video_path", "audio_path"):  # the uploads are no longer needed
                if p.get(key):
                    try:
                        Path(p[key]).unlink(missing_ok=True)
                    except Exception:
                        pass
            self._free()

    def start(self) -> None:
        worker = threading.Thread(target=self._work, name="wilrosync-worker", daemon=True)
        worker.start()
        self._threads.append(worker)
        janitor = threading.Thread(target=self._clean, name="wilrosync-janitor", daemon=True)
        janitor.start()
        self._threads.append(janitor)

    def stop(self) -> None:
        self._stop.set()

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                jid = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            job = self._jobs.get(jid)
            if job is None:
                continue
            self.run_job(job)
            self._free()

    def _clean(self) -> None:
        while not self._stop.wait(30.0):
            now = time.time()
            for job in self.jobs():
                if job.finished and now - job.finished > self.config.keep_minutes * 60:
                    if job.tmpdir is not None:
                        shutil.rmtree(job.tmpdir, ignore_errors=True)
                    with self._lock:
                        self._jobs.pop(job.id, None)


def _default_infer_fn(pipe, video: str, audio: str | None, out: str, *, bbox=None, cfg=None, **face_kwargs):
    from .pipeline.face import lipsync_face

    return lipsync_face(pipe, video, audio, out, bbox=bbox, cfg=cfg, **face_kwargs)


def _default_load_pipe(ckpt: str, config: ServerConfig):
    import torch

    from .pipeline.lipsync import LipSyncPipeline

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.dtype]
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return LipSyncPipeline.from_pretrained(
        ckpt, device=device, dtype=dtype, text_device=config.text_device, text_embeds=config.text_embeds,
        vae_dtype=torch.bfloat16 if config.vae_bf16 else torch.float32)


# ------------------------------------------------------------------------------- app
async def save_upload(f, dest: Path, what: str, max_mb: int) -> Path:
    """Write an uploaded file to ``dest``, keeping its extension; returns the path written."""
    suffix = Path(f.filename or "").suffix.lower()
    allowed = AUDIO_SUFFIXES if what == "audio" else VIDEO_SUFFIXES
    if suffix not in allowed:
        raise ValueError(f"{what} file must be one of {sorted(allowed)}, got {f.filename!r}")
    dest = dest.with_suffix(suffix)
    size = 0
    with open(dest, "wb") as w:
        while chunk := await f.read(1 << 20):
            w.write(chunk)
            size += len(chunk)
            if size > max_mb << 20:
                raise ValueError(f"{what} file is larger than {max_mb} MB")
    if not size:
        raise ValueError(f"{what} file is empty")
    return dest


def create_app(config: ServerConfig, infer_fn=None, load_pipe_fn=None):
    from . import __version__

    service = InferenceService(config, infer_fn=infer_fn, load_pipe_fn=load_pipe_fn)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.start()
        if config.ckpt and config.load_at_startup:
            threading.Thread(target=service.preload, name="wilrosync-preload", daemon=True).start()
        yield
        service.stop()

    app = FastAPI(title="wilro-sync", version=__version__, lifespan=lifespan)
    if config.cors:
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def guarded(request: Request) -> None:
        if config.api_key and request.headers.get("X-API-Key") != config.api_key:
            raise HTTPException(401, "invalid or missing X-API-Key header")

    async def read_form(request: Request) -> tuple[dict, UploadFile, UploadFile | None]:
        form = await request.form()
        data = {k: v for k, v in form.items() if not hasattr(v, "read")}
        video, audio = form.get("video"), form.get("audio")
        if not hasattr(video, "read"):
            raise HTTPException(400, "a 'video' file is required")
        return data, video, audio if hasattr(audio, "read") else None

    async def submit_job(data: dict, video: UploadFile, audio: UploadFile | None) -> Job:
        try:
            params = parse_params(data)
            if params["preview"]:
                if audio is None:
                    raise ValueError("preview also needs an 'audio' file (for its length)")
            elif audio is None:
                raise ValueError("an 'audio' file is required (only preview=true works without one)")
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        tmpdir = service.new_tmpdir()
        try:
            video_path = await save_upload(video, tmpdir / "video", "video", config.max_upload_mb)
            audio_path = await save_upload(audio, tmpdir / "audio", "audio", config.max_upload_mb) \
                if audio is not None else None
            params["video_path"] = str(video_path)
            params["audio_path"] = str(audio_path) if audio_path else None
            params["video_name"] = Path(video.filename or "out.mp4").name or "out.mp4"
            return service.submit(params, tmpdir)
        except QueueFull as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(429, str(e)) from None
        except HTTPException:
            raise
        except ValueError as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(400, str(e)) from None
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

    def send_result(job: Job):
        """The finished mp4 as a response; after it is sent, the temp files are dropped."""
        if job.status == "error":
            detail = job.error or "inference failed"
            if job.client_error and detail.startswith("ValueError: "):
                detail = detail[len("ValueError: "):]  # the exception type is noise in a 400
            raise HTTPException(400 if job.client_error else 500, detail)
        if job.delivered or job.tmpdir is None:
            raise HTTPException(410, "result already delivered once (POST the job again for a new one)")
        if job.status != "done" or job.out is None or not job.out.is_file():
            raise HTTPException(409, f"job is {job.status}; poll GET /jobs/{job.id}")
        name = Path(job.params["video_name"]).stem + "_lipsync.mp4"
        return FileResponse(job.out, media_type="video/mp4", filename=name,
                            headers={"X-Job-Id": job.id, "X-Inference-Seconds": f"{job.seconds:.1f}"},
                            background=BackgroundTask(_cleanup, job))

    @app.get("/", dependencies=[Depends(guarded)])
    def info():
        try:
            ckpt = service.resolve_ckpt(None) if (config.ckpt or config.ckpt_root) else None
        except ValueError:
            ckpt = None
        return {
            "name": "wilro-sync", "version": __version__, "ckpt": ckpt,
            "defaults": SERVE_DEFAULTS, "fields": sorted(ALLOWED_FIELDS - {"video", "audio"}),
            "endpoints": {
                "POST /lipsync": "video + audio (+ bbox, settings) -> the lip-synced mp4; "
                                 "`timeout` (s) returns 202 + job_id instead",
                "POST /jobs": "the same form as a background job -> {job_id}; "
                              "then GET /jobs/{id} and /jobs/{id}/result",
                "GET /health": "model status, device, queue depth",
            },
        }

    @app.get("/health")
    def health():
        jobs = service.jobs()
        return {
            "status": "ok", "model_loaded": service.model_loaded, "load_error": service._load_error,
            "ckpt": config.ckpt, "device": config.device or "auto",
            "jobs": {"queued": sum(j.status == "queued" for j in jobs),
                     "running": sum(j.status == "running" for j in jobs)},
        }

    @app.post("/lipsync", dependencies=[Depends(guarded)])
    async def lipsync(request: Request):
        data, video, audio = await read_form(request)
        job = await submit_job(data, video, audio)
        timeout = job.params["timeout"]
        if timeout is not None and not await _wait(job, timeout):  # too slow for this request: poll instead
            return JSONResponse({"job_id": job.id, "status": job.status,
                                 "poll": f"/jobs/{job.id}"}, status_code=202)
        await _wait(job)
        return send_result(job)

    @app.post("/jobs", dependencies=[Depends(guarded)])
    async def start_job(request: Request):
        data, video, audio = await read_form(request)
        job = await submit_job(data, video, audio)
        ahead = sum(j.status == "queued" for j in service.jobs() if j.created < job.created)
        return {"job_id": job.id, "status": job.status, "queue_position": ahead}

    @app.get("/jobs", dependencies=[Depends(guarded)])
    def list_jobs():
        return [{"job_id": j.id, "status": j.status, "inference_seconds": j.seconds} for j in service.jobs()]

    @app.get("/jobs/{jid}", dependencies=[Depends(guarded)])
    def job_status(jid: str):
        job = service.job(jid)
        if job is None:
            raise HTTPException(404, "no such job")
        return {"job_id": jid, "status": job.status, "error": job.error,
                "created_at": job.created, "started_at": job.started, "finished_at": job.finished,
                "inference_seconds": job.seconds}

    @app.get("/jobs/{jid}/result", dependencies=[Depends(guarded)])
    def job_result(jid: str):
        job = service.job(jid)
        if job is None:
            raise HTTPException(404, "no such job")
        return send_result(job)

    return app


async def _wait(job: Job, timeout: float | None = None) -> bool:
    """Wait for the job on a worker thread so the event loop stays responsive."""
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, job.event.wait, timeout)


def _cleanup(job: Job) -> None:
    """After the video has been sent: drop the temp files, keep the job record until its TTL."""
    shutil.rmtree(job.tmpdir, ignore_errors=True)
    job.tmpdir, job.delivered = None, True
