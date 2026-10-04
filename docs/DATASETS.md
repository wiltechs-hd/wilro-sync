# Training data

The OmniSync paper trains on **MEAD** (pseudo pairs, used for high-noise steps) plus about **400 hours of
in-the-wild YouTube talking-head video**. The public datasets below cover both roles.

## Recommended

| Dataset | Role in wilro-sync | Size | Resolution | Licence | How to get it |
|---|---|---|---|---|---|
| [HDTF](https://github.com/MRzzm/HDTF) | **start here**: clean, frontal, high resolution | ~16 h, 362 videos, 300+ speakers | 720p–1080p | annotations CC BY 4.0 (videos are YouTube content) | YouTube URLs, time stamps and 512×512 face-crop boxes in the repo |
| [TalkVid](https://github.com/FreedomIntelligence/TalkVid) | **main scale-up**, closest to the paper's 400 h YouTube set | 1,244 h, 7,729 speakers, 14+ languages | 480p–4K | CC BY-NC 4.0 | metadata on [Hugging Face](https://huggingface.co/datasets/FreedomIntelligence/TalkVid) + YouTube download script; ships quality scores (DOVER, CoTracker, head-detail) for filtering |
| [MEAD](https://wywu.github.io/projects/MEAD/MEAD.html) | **pseudo pairs** for timestep-dependent sampling (σ > 0.85) | 60 actors, 7 camera views, 8 emotions × 3 intensities | 1080p | research terms ([Terms of Use](https://github.com/uniBruce/Mead)) | download from the project page |
| [SpeakerVid-5M](https://huggingface.co/datasets/dorni/SpeakerVid-5M-Dataset) | later: two-person conversations for the target-aware model (milestone M5) | 5M clips | mixed | check the terms; metadata only | YouTube IDs + annotations |

Not recommended as primary data: **VoxCeleb2** (224p face crops, too low for 512 px training),
**LRS2/LRS3** (restrictive BBC/TED licences), **AVSpeech** (very noisy; needs heavy filtering).

## Suggested plan

1. **Smoke run** – 20–50 HDTF videos, ~2k steps on an L4: check that the loss falls and nothing breaks.
2. **First model** – all of HDTF + MEAD pseudo pairs + a filtered TalkVid subset (~100–200 h: high DOVER score,
   single frontal-ish face, face at least ~256 px tall).
3. **Paper scale** – grow the TalkVid subset to ~400 h.

Keep a held-out set (e.g. 20 HDTF speakers never used for training) plus a few multi-face clips for evaluation.

## Preparing the videos

* **One clearly visible speaker, good A/V sync, at least ~3 s.** `prepare_clips.py` resamples to 25 fps.
* **Face-centred crops.** `prepare_clips.py` currently centre-crops to a square, so crop landscape videos around
  the face first (HDTF's crop boxes already give 512×512 face crops; or `ffmpeg -vf crop=w:h:x:y`). Automatic
  face crops are milestone M3.
* **SyncNet filtering.** Before preparing clips, run
  `wilro-sync sync-filter --manifest wild.jsonl --root raw --out wild.synced.jsonl` (add `--assume-cropped` for
  face-crop videos). It drops videos with SyncNet confidence < 3 or |A/V offset| > 3 frames (dubbed audio, wrong
  speaker, off-screen voice, static faces) and records each kept video's offset, which `prepare_clips.py` uses to
  re-align the audio. Scores are cached in `wild.synced.report.jsonl`, so you can re-filter with other thresholds
  without re-scoring.
* **MEAD pseudo pairs**: pair two different sentences from the same actor, camera view, emotion and level, and
  write them to `raw/pairs.jsonl`:

```python
import glob, itertools, json, os, random

root = "raw"  # WORK_DIR/raw
rows = []
for d in glob.glob(f"{root}/mead/*/video/front/*/level_*"):  # actor/video/view/emotion/level
    clips = sorted(glob.glob(f"{d}/*.mp4"))
    for a, b in random.sample(list(itertools.permutations(clips, 2)), k=min(4, len(clips) * (len(clips) - 1))):
        rows.append({"kind": "pseudo_pair", "cond": os.path.relpath(a, root), "target": os.path.relpath(b, root)})
with open(f"{root}/pairs.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
```

## Storage

A prepared 512×512×49-frame clip is about **5 MB** (two latents + Whisper windows). 10k clips ≈ 50 GB, so
paper-scale data does not fit on a standard Google Drive. Use a larger Drive plan, a GCS bucket, or a cloud
VM disk for big runs.

## Licences

TalkVid (CC BY-NC) and MEAD (research terms) are non-commercial, and HDTF's annotations are CC BY but point to
YouTube videos owned by their uploaders, so models trained on this mix should be released under a non-commercial
licence. Downloading from YouTube is subject to YouTube's terms; check that your use is allowed.
