"""End-to-end smoke tests with tiny random components (CPU, seconds)."""

import numpy as np
import torch

from wilrosync.models.dit_lipsync import WanLipSyncTransformer
from wilrosync.pipeline.dscfg import DSCFGConfig
from wilrosync.pipeline.lipsync import InferenceConfig, LipSyncPipeline
from wilrosync.pipeline.target import TargetSignals
from wilrosync.train.config import load_config
from wilrosync.train.trainer import train


def test_train_smoke(tmp_path):
    cfg = load_config("configs/train/smoke_cpu.yaml", [f"output_dir={tmp_path}", "save_every=5", "keep_last=1"])
    out = train(cfg)
    assert out["steps"] == 10 and all(np.isfinite(out["losses"]))
    ckpt = tmp_path / "step_0000010"
    assert (ckpt / "wilrosync_trainable.safetensors").exists() and (ckpt / "ema").exists()
    assert not (tmp_path / "step_0000005").exists()  # keep_last pruned it
    assert (tmp_path / "log.jsonl").read_text().count("\n") == 3  # steps 1, 5, 10
    # resume continues from the saved step
    cfg2 = load_config("configs/train/smoke_cpu.yaml", [f"output_dir={tmp_path}", "max_steps=12",
                                                       f"resume_from={ckpt}"])
    assert train(cfg2)["steps"] == 12


def test_config_mixed_precision_no_unquoted():
    cfg = load_config("configs/train/smoke_cpu.yaml", ["mixed_precision=no", "data.pairs_index=null"])
    assert cfg.mixed_precision == "no" and cfg.data.pairs_index is None


def _tiny_vae():
    from diffusers import AutoencoderKLWan

    from wilrosync.models.backbone import WanVAE

    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=16, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
                           latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
    return WanVAE(vae)


class _Text:
    def __call__(self, prompts):
        return torch.randn(len(prompts), 12, 16)


class _Audio:
    def __call__(self, wav):
        n = int(np.ceil(len(wav) / 16000 * 50))
        return torch.randn(n, 3, 8)


def test_pipeline_multiface_bit_exact(tiny_base, tiny_cfg):
    model = WanLipSyncTransformer(tiny_base, tiny_cfg).eval()
    with torch.no_grad():  # make the audio path non-trivial
        for m in model.audio_attn:
            m.to_out.weight.normal_(std=0.05)
    pipe = LipSyncPipeline(model, _tiny_vae(), _Text(), _Audio(), device="cpu", dtype=torch.float32)
    rng = np.random.default_rng(0)
    t, h, w = 12, 48, 96
    frames = rng.integers(0, 255, size=(t, h, w, 3), dtype=np.uint8)
    wav = rng.standard_normal(int(16000 * 14 / 25)).astype(np.float32) * 0.1  # 14 frames of audio
    target = TargetSignals.from_bbox(t, h, w, (8, 8, 36, 40))  # left face; right half = other face
    cfg = InferenceConfig(steps=3, window=9, overlap=4, max_side=96, dscfg=DSCFGConfig(omega_peak=2.0))
    out = pipe(frames, 25.0, wav, target=target, cfg=cfg, progress=False)
    assert out.shape == (14, h, w, 3)  # follows the audio length (ping-pong extended)
    from wilrosync.io.media import pingpong_indices

    src = frames[pingpong_indices(t, 14)]
    assert np.array_equal(out[:, :, 56:], src[:, :, 56:])  # non-target region untouched
    assert not np.array_equal(out[:, 16:32, 16:28], src[:, 16:32, 16:28])  # target region generated


def test_pipeline_single_face(tiny_base, tiny_cfg):
    model = WanLipSyncTransformer(tiny_base, tiny_cfg).eval()
    pipe = LipSyncPipeline(model, _tiny_vae(), _Text(), _Audio(), device="cpu", dtype=torch.float32)
    frames = np.random.default_rng(1).integers(0, 255, size=(9, 32, 32, 3), dtype=np.uint8)
    wav = np.zeros(int(16000 * 9 / 25), dtype=np.float32)
    out = pipe(frames, 25.0, wav, cfg=InferenceConfig(steps=2, window=9, overlap=0, max_side=32), progress=False)
    assert out.shape == frames.shape
