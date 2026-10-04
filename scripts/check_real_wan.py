"""Integration check against real Wan2.1-T2V-1.3B weights (CPU is fine, ~4 GB RAM).

1. The lip-sync wrapper reproduces the pretrained transformer exactly at initialisation.
2. Real VAE round trip with Wan latent normalisation.
3. Whisper-tiny features -> audio tokens with the expected shapes.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from wilrosync.models.audio import WhisperAudioEncoder, frame_audio_windows  # noqa: E402
from wilrosync.models.backbone import WanVAE  # noqa: E402
from wilrosync.models.dit_lipsync import LipSyncModelConfig, WanLipSyncTransformer  # noqa: E402

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
torch.manual_seed(0)
dtype = torch.float32 if "--fp32" in sys.argv else torch.bfloat16

from diffusers import WanTransformer3DModel  # noqa: E402

base = WanTransformer3DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=dtype).eval()
x = torch.randn(1, 16, 2, 8, 8, dtype=dtype)
z = torch.randn(1, 16, 2, 8, 8, dtype=dtype)
t = torch.tensor([700.0])
txt = torch.randn(1, 512, 4096, dtype=dtype) * 0.1
with torch.no_grad():
    ref = base(x, t, txt, return_dict=False)[0].float()
model = WanLipSyncTransformer(base, LipSyncModelConfig()).eval()
n_new = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("audio_", "base.patch_embedding")))
print(f"total params {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B, new/expanded {n_new / 1e6:.1f}M")

whisper = WhisperAudioEncoder("openai/whisper-tiny")
wav = (0.1 * np.sin(np.linspace(0, 2000, 16000 * 2))).astype(np.float32)
feats = whisper(wav)
win = frame_audio_windows(feats, 5, 25.0, window=10)
tokens = model.encode_audio(win.unsqueeze(0))
print("whisper feats", tuple(feats.shape), "windows", tuple(win.shape), "tokens", tuple(tokens.shape))
with torch.no_grad():
    got = model(x, z, t, txt, tokens).float()
err = (got - ref).abs().max().item()
print(f"wrapper vs diffusers max abs diff: {err:.2e}")
assert err < (1e-4 if dtype == torch.float32 else 2e-2), err

vae = WanVAE.from_pretrained(REPO)
frames = torch.rand(1, 3, 5, 64, 64) * 2 - 1
lat = vae.encode(frames)
rec = vae.decode(lat)
print("latent", tuple(lat.shape), f"mean {lat.mean():.2f} std {lat.std():.2f}", "recon", tuple(rec.shape))
assert lat.shape == (1, 16, 2, 8, 8) and rec.shape == frames.shape
print("OK")
