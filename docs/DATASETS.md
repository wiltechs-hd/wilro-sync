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

## Stages

The plan is implemented as three data stages (`configs/data/*.yaml`), run with `wilro-sync data <stage>` or from
the Colab notebook (`STAGE = "..."`):

| Stage | Sources | Train data | Disk (est.) | Training |
|---|---|---|---|---|
| `smoke` | 40 HDTF segments | ~1.8 h, ~230 clips | ~4 GB | 2k steps, 1 L4/A100 |
| `first_model` | all HDTF + ~130 h TalkVid (quality-filtered, ≤ 5 min per person) + MEAD front-view pairs | ~145 h, ~4.3k identities, ~37k clips | ~370 GB | 30k steps |
| `paper_scale` | all HDTF + ~400 h TalkVid (looser filters, ≤ 10 min per person) + MEAD pairs from 3 views | ~415 h, ~7.2k identities, ~94k clips | ~1 TB | 80k steps, multi-GPU |

Estimates come from the real HDTF / TalkVid metadata; `plan` prints them for your settings and refuses to
start when they exceed `storage_budget_gb`. Every stage holds out whole identities (20 HDTF videos, ~1 % of
TalkVid speakers) for evaluation.

```bash
wilro-sync data smoke --work /data/wilro                          # plan, acquire, manifest, prepare
wilro-sync data first_model --work /data/wilro talkvid.hours=80   # any stage field can be overridden
wilro-sync data first_model --work /data/wilro --export-urls urls.txt   # videos to fetch yourself
```

What each step does:

1. **plan** – HDTF segments from its annotation files, TalkVid clips from its Hugging Face metadata filtered by
   the shipped quality scores (DOVER, head orientation/rotation, face resolution), hour budget spread
   round-robin over identities.
2. **acquire** – downloads each segment with yt-dlp (or cuts it from `raw/sources/<youtube id>.mp4` if you
   downloaded the full video yourself), then writes a **512×512, 25 fps face crop** (S3FD face track, one fixed
   square window per clip, so no crop jitter). Segments without a stable, large-enough face are dropped.
   MEAD clips (manual download, extracted to `raw/mead/`) are face-cropped the same way.
3. **manifest** – training rows (with a per-video clip count), MEAD pseudo pairs and the held-out list.
4. **prepare** – SyncNet filter (confidence ≥ 3, |offset| ≤ 3 frames, offsets corrected), then latents and audio
   windows into one clip folder shared by all stages.

Everything is resumable and shared: `first_model` reuses whatever `smoke` downloaded, cropped, scored and
prepared. Add your own (already face-cropped) videos with `extra_videos: [my_videos]` (folders under `raw/`).

**YouTube from cloud machines** (Colab, most VMs) usually hits "Sign in to confirm you're not a bot". Download
on your own computer instead, straight into Google Drive:

```bash
# on your Mac/PC (Python 3 + yt-dlp + ffmpeg; macOS: brew install yt-dlp ffmpeg), after the plan step
python3 scripts/download_segments.py \
    --plan "<Drive>/wilro-sync-work/stages/smoke/plan.json" \
    --out  "<Drive>/wilro-sync-work/raw/segments" --cookies-from-browser chrome
```

With Google Drive for desktop, `<Drive>` is a local folder (macOS: `~/Library/CloudStorage/GoogleDrive-<you>/My Drive`)
and files sync automatically; otherwise upload `raw/segments` in the Drive web UI. The acquire step then takes each
segment from `raw/segments/<segment id>.mp4` instead of YouTube (and, with `cleanup_segments`, deletes it once the
face crop exists). Only the needed time ranges are downloaded: ~3–4 GB for `smoke`, roughly 1.5 GB per hour of
video for the larger stages, so download those in batches (`--limit`) and let acquire clean up between batches.
Other options: cut from full videos you put in `raw/sources/<youtube id>.mp4`, or pass a cookies export
(`--cookies cookies.txt` / `YOUTUBE_COOKIES`) to the cloud downloader.

## Storage

A prepared 512×512×49-frame clip is about **5 MB** (two latents + Whisper windows). 10k clips ≈ 50 GB, so
paper-scale data does not fit on a standard Google Drive. Use a larger Drive plan, a GCS bucket, or a cloud
VM disk for big runs.

## Licences

TalkVid (CC BY-NC) and MEAD (research terms) are non-commercial, and HDTF's annotations are CC BY but point to
YouTube videos owned by their uploaders, so models trained on this mix should be released under a non-commercial
licence. Downloading from YouTube is subject to YouTube's terms; check that your use is allowed.
