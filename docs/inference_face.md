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
| `--window` / `--overlap` | 81 / 0 | Frames per generation window (1 + 4k) / frames blended between windows. 81 synced much better than the 49-frame training length (7.24 vs 4.84 LSE-C on xlf, 54.8k); all results below use 81. |
| `--loop` | `reverse` | Audio longer than the video: `reverse` (ping-pong), `restart`, `hold`, `trim`. |
| `--size` / `--scale` | 512 / 1.8 | Crop size and crop window relative to the face (training layout; keep). |
| `--oval` | `track` | `track`: follows the mouth. `fixed` / `fixed_old`: a static ellipse. |
| `--vae-bf16` | off | bf16 VAE: about 2× faster encode/decode, tiny colour/detail change. |
| `--profile` | off | Print time spent in VAE encode, denoising, VAE decode. |

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
