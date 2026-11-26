# HTTP server (`wilro-sync serve`)

Serve a trained checkpoint over HTTP: send a video, an audio track and an optional `bbox` (the face to
sync, in first-frame pixels); get the lip-synced video back. The settings default to the tuned
Colab-notebook values (steps=10, window=129, overlap=2, mode="lower", ...) and every field can be
overridden per request. One job runs on the GPU at a time; others wait in a queue.

```bash
pip install -e ".[server]"

# local / GPU server
wilro-sync serve --ckpt runs/first_model/step_0054800 --host 0.0.0.0 --port 8000 --api-key mysecret

# point it at a runs/ dir instead: the newest step_* inside it is used (like the notebook)
wilro-sync serve --ckpt runs/finetune --port 8000
```

Checkpoint selection: `--ckpt` may be a `step_*` directory, a `runs/<name>/` dir (its newest `step_*`
is picked), or a `step_*/ema` dir for EMA weights. With `--ckpt-root DIR` the requests may also pass
a `ckpt` field naming a checkpoint inside that dir (for A/B tests), e.g. `ckpt=step_0060500`.
Paths outside `--ckpt-root` are rejected. The model loads in the background at startup;
`GET /health` shows `model_loaded` and any `load_error`.

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | model status, device, queue depth (no API key needed) |
| `GET /` | server info: version, resolved checkpoint, the request fields, defaults |
| `POST /lipsync` | video + audio (+ fields) → the lip-synced mp4. `timeout=<s>` returns `202` + `{"job_id"}` instead of waiting |
| `POST /jobs` | the same form as a background job → `{"job_id", "queue_position"}` at once |
| `GET /jobs/{id}` | job status: `queued` → `running` → `done` / `error` (+ `error`, `inference_seconds`) |
| `GET /jobs/{id}/result` | the finished mp4 (once; kept for `--keep-minutes`, default 60, then deleted) |
| `GET /jobs` | all jobs, short form |

Errors are JSON `{"detail": ...}`: 400 bad field/file, 401 bad `X-API-Key`, 404 unknown job,
409 not finished yet, 410 result already delivered, 413 upload too large, 429 queue full,
500 inference failure.

## Request fields (multipart form)

`video` (file, required) and `audio` (file, always required — preview also uses it for its length)
plus, all optional:

| Field | Default | Values |
|---|---|---|
| `bbox` | whole frame | `x0,y0,x1,y1` in first-frame pixels; only this face is synced and tracked |
| `steps` | 10 | denoising steps (1–100) |
| `tau` | 0.92 | noise level at the start (0.01–1) |
| `window` | 129 | frames per window (1 + 4k: 49, 81, 129, ...) |
| `overlap` | 2 | frames cross-faded at window cuts |
| `chain` | 0 | window chaining: continue the last N generated frames (1 + 4k); replaces `overlap` |
| `same_noise` | off | start every window from the same noise |
| `seed` | 0 | noise seed |
| `omega_peak` / `gamma` | 4.5 / 1.5 | audio-guidance strength on the mouth / its time falloff |
| `mode` | lower | `lower`: only the mouth mask is generated (identity, eyes, nose untouched); `full`: the whole crop |
| `loop` | reverse | audio longer than video: `reverse` (ping-pong), `restart`, `hold`, `trim` |
| `pause` | 0 | extra held frames at each loop turn |
| `crop` | auto | `auto` / `fixed` / `track` (lazy camera for fast head motion) |
| `crop_smooth`, `crop_deadzone` | 9, 0.15 | tracked window smoothing / allowed drift |
| `preroll`, `postroll` | 4, 8 | frames generated before/after and dropped (fix cold first/last frames); 0 = off |
| `reference` | stride2 | frames the model is conditioned on: `stride2`, `same`, `shift`, `reverse` |
| `blend` | double | how the mouth is blended in (`double` is the evaluated default) |
| `oval` | track | `track`, `fixed`, `fixed_old`, or a JSON ellipse `{"cx":..,"cy":..,"rx":..,"ry":..,"feather":..}` |
| `track` | 1.8/0.2/0.1/0.2 | JSON for the tracked oval shape `{"width","bottom","top","feather","spread","spread_up","smooth"}` |
| `size`, `scale` | 512, 1.8 | crop size and crop window / face size (keep the training layout) |
| `preview` | off | `true`: write the crop with the mask drawn, no model (seconds; check `bbox` first) |
| `det_every` | auto | face detection every n-th frame |
| `timeout` | wait forever | `POST /lipsync` only: seconds to wait before returning `202` + job_id (0 = return at once) |
| `ckpt` | server default | checkpoint inside `--ckpt-root` |

The defaults are the notebook's tuned `BASE` (`steps=10, window=129, overlap=2, tau=0.92, mode=lower,
loop=reverse, crop=auto, preroll=4, postroll=8, reference=stride2, blend=double, oval=track` with
`track = width 1.8, bottom 0.2, top 0.1, spread 0.2`, `omega_peak 4.5, gamma 1.5`).

## Examples

Sync (simplest; hold the connection until the video is ready):

```bash
curl -X POST http://localhost:8000/lipsync \
  -H "X-API-Key: mysecret" \
  -F video=@in.mp4 -F audio=@speech.wav \
  -F bbox=575,140,675,255 -F timeout=900 \
  -o out.mp4
```

```python
import requests

r = requests.post("http://localhost:8000/lipsync",
                  headers={"X-API-Key": "mysecret"},
                  files={"video": open("in.mp4", "rb"), "audio": open("speech.wav", "rb")},
                  data={"bbox": "575,140,675,255", "timeout": 900}, timeout=960)
r.raise_for_status()
open("out.mp4", "wb").write(r.content)
```

Background job (robust behind proxies; the right way on Colab):

```python
import time, requests

BASE, KEY = "http://localhost:8000", {"X-API-Key": "mysecret"}
job = requests.post(f"{BASE}/jobs", headers=KEY,
                    files={"video": open("in.mp4", "rb"), "audio": open("speech.wav", "rb")},
                    data={"bbox": "410,80,620,330"}).json()
while True:
    s = requests.get(f"{BASE}/jobs/{job['job_id']}", headers=KEY).json()
    if s["status"] in ("done", "error"):
        break
    time.sleep(3)
if s["status"] == "done":
    open("out.mp4", "wb").write(requests.get(f"{BASE}/jobs/{job['job_id']}/result", headers=KEY).content)
else:
    print(s["error"])
```

Check the target box first (no model, no GPU, returns in seconds):

```bash
curl -X POST http://localhost:8000/lipsync -F video=@in.mp4 -F audio=@a.wav \
     -F bbox=575,140,675,255 -F preview=true -o mask.mp4
```

## Client script

`scripts/server_client.py` is a stdlib-only client (no `requests` needed) covering all of the above:
`--health`, a blocking call, `--bg` job + polling, and `--preview`; extra pipeline settings go as
`FIELD=VALUE` pairs, and the URL/key can come from `WILROSYNC_URL` / `WILROSYNC_KEY`:

```bash
python scripts/server_client.py http://192.168.1.103:8000 in.mp4 speech.wav \
    --bbox 575,140,675,255 --bg -o out.mp4 --key mysecret
python scripts/server_client.py http://192.168.1.103:8000 --health
```

## Colab

```python
# cell 1: start the server from the notebook (its checkpoint on Drive)
!pip install -q -e ".[server]"
import threading
threading.Thread(
    target=lambda: __import__("subprocess").run(
        ["wilro-sync", "serve", "--ckpt", f"{WORK_DIR}/runs/finetune/step_0060500",
         "--host", "127.0.0.1", "--port", "8000"], check=True),
    daemon=True).start()

# cell 2: a public HTTPS URL (or use pyngrok)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared && ./cloudflared tunnel --url http://localhost:8000 &> tunnel.log &
!sleep 4 && grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log
```

Notebook cell 2 of this doc's examples then posts to the printed `https://...trycloudflare.com` URL.
Proxies time long connections out, so prefer `POST /jobs` + `GET /jobs/{id}` over a blocking
`POST /lipsync` there.

## Notes

- One GPU job runs at a time; `--max-queue` (default 8) caps how many may wait (extra → 429).
- The server is single-process (`workers=1`): one model copy on the GPU. Scale by starting more
  servers on more GPUs / machines, not more workers.
- Files are processed in temp dirs; uploads are deleted right after inference, results are kept for
  `--keep-minutes` (default 60) after the job finishes. Nothing is written into the repo.
- Inference needs the base Wan/Whisper/umT5 weights in the HF cache: run `huggingface-cli login`
  (or set `HF_TOKEN`) and, for the SyncNet/face weights, nothing extra — they download into
  `$WILROSYNC_CACHE` on first use. umT5 (~11 GB) loads only to encode the prompt, then frees.
- `--device` defaults to cuda when available, else cpu. `--dtype bf16` (default) matches the notebook.
- Security: bind to 127.0.0.1 and put a proxy in front, or set `--api-key`; there is no HTTPS. `--cors`
  allows browser apps to call the API directly. Treat uploads as untrusted input: the server only
  writes them to temp files and hands them to ffmpeg/imageio.

## Small-memory devices (Jetson)

The text encoder (umT5, about 11 GB) is only used to encode the prompt, which never changes. On a device where
CPU and GPU share 16 GB or less (e.g. Jetson Orin NX), loading it next to the model runs out of memory. Encode
the prompt once while nothing else is running, then start the server with the saved file; the text encoder is
then never loaded (requests with a different `prompt` get a 400):

```bash
wilro-sync encode-prompt --ckpt ~/wilro/ckpt/step_0057000 --out ~/wilro/prompt.safetensors    # once, on the CPU
wilro-sync serve --ckpt ~/wilro/ckpt/step_0057000 --text-embeds ~/wilro/prompt.safetensors \
    --host 0.0.0.0 --port 8000 --api-key <key> --vae-bf16
```

The file can also be made on another machine (it is a few MB) and copied over. After encoding, the text encoder
can be removed from the Hugging Face cache to free about 11 GB of disk (the rest of the base model is still
needed). `wilro-sync infer-face` takes the same `--text-embeds` option.

Video memory: the source frames are decoded to a temporary file in the system temp folder (about
width x height x 3 bytes per frame, e.g. 1.5 GB for 10 s of 1080p) and read from there, and the result is written
in small chunks, so RAM use depends on the 512 px face crop, not on the size or length of the video. Make sure the
temp folder (usually `/tmp`) has that much free disk.

GPUs with less than 24 GB (and the shared memory of a Jetson) automatically run the with- and without-audio
guidance passes one after the other instead of as a batch of 2 (`InferenceConfig.cfg_sequential`): about half the
transformer's working memory, a little slower, same result. After loading and after every job the server hands
freed memory back to the system (on a Jetson the CPU heap left over from loading is otherwise 3-4 GB).
