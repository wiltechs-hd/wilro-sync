# wilro-sync

Mask-free, diffusion-transformer lip sync that only moves the face you choose.

wilro-sync is an independent, open-source implementation of
[**OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers**](https://arxiv.org/abs/2505.21448)
(Peng et al., NeurIPS 2025) on top of the open [Wan2.x](https://github.com/Wan-Video/Wan2.2) video DiT. It
adds **target-face control**: in multi-face scenes only the selected face is lip-synced, and every other pixel
is kept bit-exact.

> **Status: pre-alpha.** The model, training and inference code is in place and tested on CPU with tiny
> random models and against the real Wan2.1-1.3B weights. **No trained checkpoint has been released yet.**
> See [docs/PLAN.md](docs/PLAN.md) for the roadmap.

![architecture](docs/assets/architecture.svg)

## How it works

| Piece | Paper | Where |
|---|---|---|
| Mask-free conditional flow matching `(V_cd, A_ab) → V_ab` | §3.2 | `wilrosync/train/trainer.py` |
| Source latent channel-concatenated with the noisy latent | Fig. 2 | `wilrosync/models/dit_lipsync.py` |
| Audio cross-attention in every DiT block (Whisper features) | Fig. 2 | `wilrosync/models/audio*.py` |
| Timestep-dependent sampling (pseudo pairs for σ > 0.85) | §3.2, Eq. 3 | `wilrosync/data/datasets.py` |
| Progressive noise initialisation (τ = 0.92) | §3.3 | `wilrosync/flow.py`, `pipeline/lipsync.py` |
| Dynamic spatiotemporal CFG (γ = 1.5) | §3.4 | `wilrosync/pipeline/dscfg.py` |
| **Target-focused guidance, region lock, pixel composite** | new | `wilrosync/pipeline/{dscfg,region_lock,composite}.py` |

The Wan transformer weights are reused unchanged: new input channels and the audio attention output
projections start at zero, so at initialisation the model computes exactly what Wan computes (verified
against the real 1.3B weights by `scripts/check_real_wan.py`).

## Install

```bash
git clone https://github.com/wiltechs-hd/wilro-sync && cd wilro-sync
pip install -e ".[dev]"          # add ".[train]" for 8-bit Adam
pytest -q                        # CPU, ~10 s
```

## Google Colab

[`notebooks/wilro_sync_colab.ipynb`](notebooks/wilro_sync_colab.ipynb) walks through setup, sanity checks, data
preparation, training (L4 / A100) and inference in plain Python (no form widgets), keeping clips and checkpoints on Google Drive so a disconnected
session can resume.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wiltechs-hd/wilro-sync/blob/main/notebooks/wilro_sync_colab.ipynb)

## Inference

```bash
# single speaker
wilro-sync infer --ckpt runs/stage_a/step_0080000 --video in.mp4 --audio speech.wav --out out.mp4

# multi-face scene: only the face inside the box is synced
wilro-sync infer --ckpt ... --video two_people.mp4 --audio speech.wav --out out.mp4 --bbox 410,80,620,330
```

Options: `--config configs/infer/default.yaml`, `--steps`, `--tau`, `--omega-peak`, `--gamma`,
`--max-side`, `--text-device cpu` (keeps the 11 GB umT5 encoder off the GPU), `--no-region-lock`.
Detector-based target selection (reference image, track id, active speaker) is milestone M1.

## Training

Which datasets to use and how to prepare them: [docs/DATASETS.md](docs/DATASETS.md) (HDTF to start, TalkVid to
scale, MEAD for pseudo pairs).

```bash
# 1. pre-compute latents, Whisper windows and caption embeddings (GPU recommended)
python scripts/prepare_clips.py --manifest data/mead_pairs.jsonl --root data/raw --out data/latents/mead
python scripts/prepare_clips.py --manifest data/wild.jsonl --root data/raw --out data/latents/wild

# 2. train (single GPU, or `accelerate launch -m wilrosync.cli train ...` for multi-GPU)
wilro-sync train --config configs/train/stage_a.yaml
wilro-sync train --config configs/train/stage_a.yaml train_mode=adapter   # ablation: no LoRA

# smoke test with a tiny random model on CPU
wilro-sync train --config configs/train/smoke_cpu.yaml
```

Rough memory for Wan2.1-1.3B + audio adapter + LoRA at 512×512×49 frames: ~16–24 GB (estimate).
The frozen VAE, Whisper and umT5 run only in `prepare_clips.py`, never during training.

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{peng2025omnisync,
  title     = {OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers},
  author    = {Peng, Ziqiao and Liu, Jiwen and Zhang, Haoxian and Liu, Xiaoqiang and Tang, Songlin and
               Wan, Pengfei and Zhang, Di and Liu, Hongyan and He, Jun},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

## Responsible use

Lip-sync models can be misused to put words in people's mouths. Only process footage you have the rights
and consent to edit, label synthetic media as such, and do not use this project to impersonate real people.

## License

Code: Apache-2.0. Model weights (when released) may carry additional restrictions inherited from the
training datasets (e.g. MEAD, VoxCeleb2 and HDTF are research-only).
