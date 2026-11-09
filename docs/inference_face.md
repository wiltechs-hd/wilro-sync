# Face-crop inference (`infer-face`)

The recommended way to run a trained model. The target face is cropped exactly as in training (a fixed
square, 1.8 × the face, resized to 512), synced, and pasted back. In `lower` mode only a mouth mask is
generated; the rest of the face stays the original video.

```bash
pip install -e ".[face]"          # MediaPipe, for the tracked oval

# best settings by default
wilro-sync infer-face --ckpt runs/first_model/step_0054800 --video in.mp4 --audio speech.wav --out out.mp4

# check the mask first (seconds, no model): green = generated, red = soft edge, rest = original
wilro-sync infer-face --video in.mp4 --out mask.mp4 --preview

# multi-face scene: the face inside the box (first frame) is synced
wilro-sync infer-face --ckpt ... --video two.mp4 --audio speech.wav --out out.mp4 --bbox 410,80,620,330
```

Python:

```python
from wilrosync.pipeline.lipsync import LipSyncPipeline
from wilrosync.pipeline.face import face_infer_config, lipsync_face

pipe = LipSyncPipeline.from_pretrained("runs/first_model/step_0054800")   # LoRA merged for speed
lipsync_face(pipe, "in.mp4", "speech.wav", "out.mp4",
             track=dict(width=2.2, bottom=0.4, top=0.3),                    # = the defaults
             cfg=face_infer_config(steps=10, tau=0.92))                    # = the defaults
```

The weights for face detection and SyncNet download on first use into `$WILROSYNC_CACHE`
(default `~/.cache/wilrosync`). If the original Oxford server is down they come from the Hugging Face mirror.

## Parameters

| Option | Default | What it does |
|---|---|---|
| `--mode` | `lower` | `lower`: only the mouth mask is generated (keeps identity, eyes, nose). `full`: the whole crop is generated (like training; more freedom, identity can drift). |
| `--steps` | 10 | Denoising steps. 10 synced better than both 5 and 20 (see results). |
| `--tau` | 0.92 | Noise level at the start (progressive noise init). |
| `--omega-peak` | 4.5 | Audio guidance strength on the mouth. |
| `--window` / `--overlap` | 129 / 0 | Frames per generation window (1 + 4k) / frames cross-faded at each cut. Longer windows sync better and have fewer cuts: unseen LSE-C average at 55.5k 7.26 (81), 7.46 (97), 7.59 (129), 10 GB peak on an L4 at 129. Each window is generated separately, so the mouth can look slightly different after a cut; `--overlap 8` to `16` fades over that many frames (longer fades can mix two mouth shapes). The fade always covers exactly `--overlap` frames, also at the last window (shifted back to end on the last frame); shared frames before the fade come from the earlier window. `0` = hard cut. |
| `--same-noise` | off | Start every window from the same noise instead of fresh noise, which tends to keep teeth and lip texture more alike from window to window. |
| `--chain` | 0 | Window chaining (0 = off). Each window after the first starts from the last N frames already generated (1 + 4k, e.g. 9), held fixed, so it continues them instead of starting cold; nothing is cross-faded. Fixes a sudden change in sharpness or look at window cuts (every 81 frames, shifted by `--preroll`). About 7% slower at 5, 12% at 9. Off by default: on the 5-clip test at 55.5k it lowered the unseen LSE-C average from 7.48 to 7.22 (5 and 9 alike). Use instead of `--overlap`. |
| `--loop` | `reverse` | Audio longer than the video: `reverse` (ping-pong), `restart`, `hold`, `trim`. |
| `--size` / `--scale` | 512 / 1.8 | Crop size and crop window relative to the face (training layout; keep). |
| `--crop` | `auto` | `auto` (default): the fixed window when the head moves less than half the face size over the clip, otherwise `track`. `fixed`: one crop window for the whole clip. `track`: the window follows the face like a lazy camera (still while the face stays within `--crop-deadzone` 0.15 × face of it, then moves just enough; smoothed over `--crop-smooth` 9 frames). |
| `--oval` | `track` | `track`: follows the mouth. `fixed` / `fixed_old`: a static ellipse. |
| `--preroll` | 0 | Frames generated before the start and dropped afterwards (the first frames played backwards, with silence), so the first real frames aren't generated cold. Try 8 (0.3 s) when the very first frames of a video look bad. |
| `--blend` | `double` | `double`: the mouth is blended in the crop (mask + 6 px feather) and again when pasted, so its weight is about mask², and the source mouth shows through more in the soft edge. `single`: blended once, exactly the mask `--preview` shows. Try `single` if some frames show a doubled lip or mouth outline. |
| `--raw-out` | none | Also write the crops as generated, before any blending (source crop \| generated), to see whether an artefact comes from the model or from the blending. |
| `--vae-bf16` | off | bf16 VAE: about 2× faster encode/decode, tiny colour/detail change. |
| `--profile` | off | Print time spent in VAE encode, denoising, VAE decode. |

### Fast head motion

With `--crop fixed` the window must cover every position of the head, so a moving face ends up small and off-centre
in the 512 crop (the warning `face moves 0.7x its size` points to this). `--crop track` keeps a constant window of
1.8 × the face that follows it and pastes each frame back at its own position.

Measured (54.8k, LSE-C, fixed → first `track` version that re-centred on every frame): demo2 (moves 0.7× its size)
6.01 → 7.05; xlf (0.1×) 7.26 → 7.25; demo1 (0.2×) 7.12 → 5.95; good_man (0.4×) 6.59 → 6.08. Re-centring on every
frame passes detector jitter to the whole crop, which hurts still clips, so `track` now moves like a lazy camera
(dead zone) and `auto` keeps the fixed window for clips that move less than half the face size.

Lazy-camera `track` and `auto` on the same four clips (54.8k):

| Clip | Head motion | fixed | track (every frame) | track (lazy camera) | **auto** |
|---|---|---|---|---|---|
| demo2 | 0.7 × face | 6.01 | 7.05 | 6.93 | **6.93** (track) |
| xlf | 0.1 × face | 7.26 | 7.25 | 6.80 | **7.26** (fixed) |
| demo1 | 0.2 × face | 7.12 | 5.95 | 6.76 | **7.12** (fixed) |
| good_man (training clip) | 0.4 × face | 6.59 | 6.08 | 6.83 | **6.59** (fixed) |

`auto` is the default: it matches the better crop on every unseen clip. Even a still window of a slightly different
size and position (the lazy camera did not move on xlf and demo1) changed LSE-C by about 0.4, so small crop
differences on one clip are close to noise; average over clips (or seeds) before tuning `--crop-deadzone`. The face tracker also
follows fast motion now: when the box overlap with the previous detection drops below 0.3, it takes the nearest
face of similar size within one face width instead of losing the track.

### Tracked oval

Position and head tilt follow MediaPipe face landmarks every frame (smoothed over 5 frames); the size is fixed
for the clip (widest mouth, widest jaw opening), so the mask never pulses. Shape: a flat top between the nose and
the upper lip, a round bottom below the chin.

| Option | Default | Meaning | Effect in tests |
|---|---|---|---|
| `--bottom` | 0.4 | Room below the chin, on top of the source's widest jaw opening (× nose-to-chin distance). | **The one that matters**: 0.1 → 0.3 gave about +0.9 LSE-C on a clip with little source mouth motion. The chin must be free to drop when the new audio needs a wider mouth. |
| `--width` | 2.2 | Half-width = this × half the widest mouth width. | 2.1 to 2.7: within noise. |
| `--top` | 0.3 | Top edge: 0 = bottom of the nose, 1 = top of the upper lip. The soft edge is capped to stay between the two (no double lips). | Small effect; 0.3 lets the philtrum move. |
| `--spread` | 0 | Extra-wide soft edge on the sides and below. | No effect on sync. |
| `--spread-up` | 0 | How high the spread may reach: 0 = nose, 1 = just under the eyes. Needs `--spread` ≥ 0.3 to reach the nose. | Neutral on sync; a visual choice if a still nose looks unnatural (`--spread 0.3 --spread-up 1`). Watch for a double nostril outline. |

## Results (SyncNet LSE-C, higher is better)

Same audio for every run; averages over two checkpoints 200 steps apart (54.6k / 54.8k), because neighbouring
checkpoints differ by up to ±0.3 to 0.4. `xlf` is unseen; `good_man` is in the training data (it shows
memorisation more than quality). Differences under about 0.1 are noise.

| Setting | xlf (unseen) | good_man (training clip) |
|---|---|---|
| tracked w2.2 b0.4 t0.3, 10 steps | 7.19 | 7.64 |
| same with spread 0.3, spread_up 1 | 7.21 | 7.54 |
| fixed_old ellipse, 10 steps | 7.21 | 7.41 |
| tracked w2.2 b0.1 t0.3 (little room below the chin), 10 steps | 7.04 | 6.68 |
| tracked ovals at 5 steps (average of 4) | 6.97 | 6.60 |
| same 11 ovals at 20 steps / at 10 steps | 6.79 / 7.06 | 5.65 / 6.85 |

## Speed

Per 81-frame window (about 3 s of video), each step runs the transformer once on two copies of the latent (with
and without audio, for guidance), so 10 steps = 20 transformer passes; the VAE encodes and decodes once.
Built in:

* LoRA merged into the weights at load time (`merge_lora=True`): same output, faster.
  Checked on xlf at 54.8k: 7.24 unmerged, 7.22 merged (noise).
* The no-audio copy is skipped in steps where guidance can no longer change the mouth
  (`omega_peak · σ^γ ≤ min_scale`, the last 2 of 10 steps at the defaults): identical inside the mask, only the
  rim of the region lock differs (xlf at 54.8k: 7.24 without, 7.29 with). `InferenceConfig.skip_uncond` (on),
  `cfg_stop_sigma` (0 = only where exact).
