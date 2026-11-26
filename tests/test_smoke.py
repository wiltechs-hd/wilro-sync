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


def test_resume_applies_new_lr_and_decay(tmp_path):
    import json

    from wilrosync.train.trainer import _lr_lambda

    cfg = load_config("configs/train/smoke_cpu.yaml", [f"output_dir={tmp_path}", "save_every=10"])
    train(cfg)
    # resume with a third of the learning rates and a cosine decay from the resume step
    cfg2 = load_config("configs/train/smoke_cpu.yaml", [
        f"output_dir={tmp_path}", f"resume_from={tmp_path / 'step_0000010'}", "max_steps=30", "log_every=1",
        f"lr={cfg.lr / 3}", f"lr_new={cfg.lr_new / 3}", "lr_decay_start=10", "lr_min_ratio=0.1"])
    train(cfg2)
    logs = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    after = {r["step"]: r["lr"] for r in logs if r["step"] > 10}
    f = _lr_lambda(cfg2)
    # the logged lr is the first group's (new layers): lr_new / 3, decayed
    assert abs(after[11] - cfg.lr_new / 3 * f(11)) / (cfg.lr_new / 3) < 0.05
    assert after[30] < after[11] and abs(after[30] - cfg.lr_new / 3 * 0.1) / (cfg.lr_new / 3) < 0.05


def test_lr_lambda_shape():
    from wilrosync.train.trainer import _lr_lambda

    cfg = load_config("configs/train/smoke_cpu.yaml", ["warmup_steps=0", "max_steps=100", "lr_decay_start=50",
                                                       "lr_min_ratio=0.2"])
    f = _lr_lambda(cfg)
    assert f(10) == 1.0 and f(50) == 1.0 and abs(f(75) - 0.6) < 1e-6 and abs(f(100) - 0.2) < 1e-6


def test_pipeline_chained_windows(tiny_base, tiny_cfg):
    model = WanLipSyncTransformer(tiny_base, tiny_cfg).eval()
    pipe = LipSyncPipeline(model, _tiny_vae(), _Text(), _Audio(), device="cpu", dtype=torch.float32)
    rng = np.random.default_rng(2)
    t, h, w = 30, 32, 32
    frames = rng.integers(0, 255, size=(t, h, w, 3), dtype=np.uint8)
    wav = rng.standard_normal(int(16000 * t / 25)).astype(np.float32) * 0.1
    target = TargetSignals.from_bbox(t, h, w, (4, 4, 28, 28))
    runs, locks = [], []
    run, denoise = pipe._run_window, pipe._denoise_window

    def spy_run(*a, **k):
        out = run(*a, **k)
        runs.append((a[1], k.get("ctx_px"), k.get("n_lock", 0), out))
        return out

    def spy_denoise(z_src, *a, **k):
        x = denoise(z_src, *a, **k)
        locks.append((z_src, x))
        return x

    pipe._run_window, pipe._denoise_window = spy_run, spy_denoise
    cfg = InferenceConfig(steps=3, window=13, overlap=0, chain=5, max_side=32, loop_video=False)
    out = pipe(frames, 25.0, wav, target=target, cfg=cfg, progress=False)
    assert out.shape == (t, h, w, 3)
    assert runs[0][1] is None and all(r[2] >= 2 for r in runs[1:])  # >= 5 context frames = 2 latent frames
    for (w0, _, _, d0), (w1, ctx, _, _) in zip(runs, runs[1:]):
        if w1.start == w0.start + 8:  # regular stride: the context is the previous window's last 5 frames
            assert torch.allclose(ctx, d0[w1.start - w0.start: w1.start - w0.start + 5])
    for (z_src, x), (_, _, n_lock, _) in zip(locks, runs):
        assert torch.allclose(x[:, :, :n_lock], z_src[:, :, :n_lock], atol=1e-5)  # context held fixed
    try:
        pipe(frames, 25.0, wav, target=target, cfg=InferenceConfig(steps=1, window=9, overlap=2, chain=5,
                                                                    max_side=32), progress=False)
        raise AssertionError("chain + overlap must be rejected")
    except ValueError:
        pass


def test_lpips_loss_smoke(tmp_path, monkeypatch):
    import torch.nn as nn

    from wilrosync.flow import add_noise, target_velocity
    from wilrosync.train import perceptual

    # the one-step guess of the clean latents is exact for the true velocity
    z, eps, s = torch.randn(2, 4, 3, 8, 8), torch.randn(2, 4, 3, 8, 8), torch.tensor([0.2, 0.7])
    x_t = add_noise(z, eps, s)
    torch.testing.assert_close(x_t - s.view(-1, 1, 1, 1, 1) * target_velocity(z, eps), z, atol=1e-5, rtol=0)
    # crops stay inside the latent frame
    mouth = torch.tensor([[[0.95, 0.95]] * 3, [[0.5, 0.6]] * 3])
    assert perceptual.mouth_crop_boxes(mouth, (8, 8), 4) == [(4, 4, 4), (3, 2, 4)]

    calls = []

    class _FakeLPIPS(nn.Module):  # stands in for the pretrained network (no download in tests)
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1).requires_grad_(False)

        def forward(self, a, b):
            calls.append(a.shape)
            assert a.requires_grad and not b.requires_grad
            return (self.conv(a) - self.conv(b)).abs().mean(dim=(1, 2, 3), keepdim=True)

    monkeypatch.setattr(perceptual, "make_lpips", lambda net="alex": _FakeLPIPS())
    cfg = load_config("configs/train/smoke_cpu.yaml", [f"output_dir={tmp_path}", "max_steps=3", "log_every=1",
                                                      "lpips_weight=1.0", "lpips_max_sigma=1.1", "lpips_crop=4"])
    out = train(cfg)
    assert out["steps"] == 3 and all(np.isfinite(out["losses"]))
    assert calls and calls[0][-1] == 4 * 8  # a 4-cell crop decodes to 32 px
    import json

    logs = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert all("lpips" in r for r in logs)


def test_resume_restarts_ema_from_the_resumed_weights(tmp_path):
    from safetensors.torch import load_file

    cfg = load_config("configs/train/smoke_cpu.yaml", [f"output_dir={tmp_path}", "save_every=10", "lr=0.05",
                                                      "lr_new=0.05"])
    train(cfg)
    raw10 = load_file(str(tmp_path / "step_0000010" / "wilrosync_trainable.safetensors"))
    cfg2 = load_config("configs/train/smoke_cpu.yaml", [f"output_dir={tmp_path}", "max_steps=11", "lr=0.0",
                                                       "lr_new=0.0", f"resume_from={tmp_path / 'step_0000010'}"])
    train(cfg2)
    ema11 = load_file(str(tmp_path / "step_0000011" / "ema" / "wilrosync_trainable.safetensors"))
    # with the learning rate at 0 the weights stay at step 10, so a correctly restarted EMA equals them
    for k, v in raw10.items():
        torch.testing.assert_close(ema11[k].float(), v.float(), atol=1e-5, rtol=1e-4)


def test_pipeline_reference_conditions_the_model(tiny_base, tiny_cfg):
    model = WanLipSyncTransformer(tiny_base, tiny_cfg).eval()
    pipe = LipSyncPipeline(model, _tiny_vae(), _Text(), _Audio(), device="cpu", dtype=torch.float32)
    rng = np.random.default_rng(3)
    t, h, w = 9, 32, 32
    frames = rng.integers(0, 255, size=(t, h, w, 3), dtype=np.uint8)
    wav = np.zeros(int(16000 * t / 25), dtype=np.float32)
    target = TargetSignals.from_bbox(t, h, w, (4, 4, 28, 28))
    conds = []
    fwd = model.forward

    def spy(x_t, z_cond, *a, **k):
        conds.append(z_cond.detach().clone())
        return fwd(x_t, z_cond, *a, **k)

    model.forward = spy
    cfg = InferenceConfig(steps=2, window=9, overlap=0, max_side=32, loop_video=False)
    out_same = pipe(frames, 25.0, wav, target=target, cfg=cfg, progress=False)
    n_same = len(conds)
    out_ref = pipe(frames, 25.0, wav, target=target, cfg=cfg, progress=False, ref_frames=frames[::-1].copy())
    assert not torch.equal(conds[0], conds[n_same])  # the model saw a different reference
    assert out_same.shape == out_ref.shape
    assert np.array_equal(out_same[:, :2], out_ref[:, :2])  # outside the target: still the real frames


def test_saved_prompt_embeds(tmp_path, tiny_base, tiny_cfg):
    import json

    import pytest

    from wilrosync.models import backbone
    from wilrosync.models.dit_lipsync import CONFIG_NAME
    from wilrosync.pipeline import lipsync as lp

    emb = torch.randn(1, 12, 16)
    path = str(tmp_path / "prompt.safetensors")
    LipSyncPipeline.save_prompt_embeds(path, "hello", emb)
    model = WanLipSyncTransformer(tiny_base, tiny_cfg).eval()
    pipe = LipSyncPipeline(model, None, None, None, device="cpu", dtype=torch.float32)
    assert pipe.load_prompt_embeds(path) == "hello"
    torch.testing.assert_close(pipe.encode_prompt("hello"), emb)
    with pytest.raises(ValueError, match="not in the saved prompt"):
        pipe.encode_prompt("another prompt")  # no text encoder: refused instead of loading umT5

    class _FakeT5:  # encode-prompt loads only the text encoder named by the checkpoint
        calls = []

        @classmethod
        def from_pretrained(cls, repo, torch_dtype=None, device=None):
            cls.calls.append((repo, device))
            return lambda prompts: torch.ones(len(prompts), 3, 4)

    ckpt = tmp_path / "step_1"
    ckpt.mkdir()
    (ckpt / CONFIG_NAME).write_text(json.dumps({"base_repo": "some/repo"}))
    orig = backbone.T5TextEncoder
    backbone.T5TextEncoder = _FakeT5
    try:
        out = lp.encode_prompt_file(str(ckpt), str(tmp_path / "p.safetensors"), "hi", device="cpu")
    finally:
        backbone.T5TextEncoder = orig
    assert _FakeT5.calls == [("some/repo", "cpu")]
    pipe2 = LipSyncPipeline(model, None, None, None, device="cpu", dtype=torch.float32)
    assert pipe2.load_prompt_embeds(out) == "hi"


def test_sequential_guidance_matches_batched(tiny_base, tiny_cfg):
    model = WanLipSyncTransformer(tiny_base, tiny_cfg).eval()
    with torch.no_grad():  # make the audio path non-trivial so guidance matters
        for m in model.audio_attn:
            m.to_out.weight.normal_(std=0.05)
    pipe = LipSyncPipeline(model, _tiny_vae(), _Text(), _Audio(), device="cpu", dtype=torch.float32)
    rng = np.random.default_rng(4)
    t, h, w = 9, 32, 32
    frames = rng.integers(0, 255, size=(t, h, w, 3), dtype=np.uint8)
    wav = rng.standard_normal(int(16000 * t / 25)).astype(np.float32) * 0.1
    target = TargetSignals.from_bbox(t, h, w, (4, 4, 28, 28))
    outs = []
    for seq in (False, True):
        torch.manual_seed(0)
        cfg = InferenceConfig(steps=3, window=9, overlap=0, max_side=32, loop_video=False, cfg_sequential=seq,
                              skip_uncond=False, dscfg=DSCFGConfig(omega_peak=3.0))
        outs.append(pipe(frames, 25.0, wav, target=target, cfg=cfg, progress=False).astype(int))
    assert np.abs(outs[0] - outs[1]).max() <= 1  # same result, one pass at a time


def test_release_memory_runs():
    from wilrosync.pipeline.lipsync import release_memory

    release_memory()


def test_from_checkpoint_loads_the_base_on_the_device(tmp_path, tiny_base, tiny_cfg, monkeypatch):
    model = WanLipSyncTransformer(tiny_base, tiny_cfg)
    model.save_checkpoint(str(tmp_path))
    seen = []

    def fake_from_base(cls, cfg, torch_dtype=torch.bfloat16, **kw):
        seen.append(kw)
        return cls(tiny_base, cfg)

    monkeypatch.setattr(WanLipSyncTransformer, "from_base", classmethod(fake_from_base))
    try:
        WanLipSyncTransformer.from_checkpoint(str(tmp_path), torch_dtype=torch.float32, device="cuda")
    except (AssertionError, RuntimeError):  # moving the result to cuda fails without a GPU
        pass
    assert seen and seen[-1] == {"device_map": "cuda"}
    WanLipSyncTransformer.from_checkpoint(str(tmp_path), torch_dtype=torch.float32, device="cpu")
    assert seen[-1] == {}  # CPU: plain loading
