"""Face-crop inference: loops, tracked oval masks, no-audio-pass skipping, LoRA merge, end-to-end (CPU)."""

import numpy as np
import pytest
import torch

from wilrosync.pipeline.dscfg import DSCFGConfig
from wilrosync.pipeline.face import face_infer_config, lipsync_face, loop_indices
from wilrosync.pipeline.lipsync import InferenceConfig, LipSyncPipeline
from wilrosync.pipeline.oval import TrackConfig, lower_face_mask, oval_from_landmarks

# synthetic face in crop units: eyeL, eyeR, mouthL, mouthR, subnasale, chin, lip top
FACE = np.array([[0.35, 0.40], [0.65, 0.40], [0.40, 0.62], [0.60, 0.62], [0.50, 0.52], [0.50, 0.80],
                 [0.50, 0.57]], np.float32)


def _landmarks(t=20, shift=0.0, jaw=0.0):
    lm = np.repeat(FACE[None], t, 0).copy()
    lm[:, :, 0] += shift
    lm[:, 5, 1] += jaw * np.sin(np.arange(t) / 2) ** 2  # chin drops when the mouth opens
    return lm


def _at(mask, xy, size):
    return float(mask[int(xy[1] * size), int(xy[0] * size)])


# ----------------------------------------------------------------------------- loops
def test_loop_indices():
    assert loop_indices(4, 9, "reverse").tolist() == [0, 1, 2, 3, 2, 1, 0, 1, 2]
    assert loop_indices(4, 9, "restart").tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0]
    assert loop_indices(4, 6, "hold").tolist() == [0, 1, 2, 3, 3, 3]
    assert loop_indices(4, 9, "trim").tolist() == [0, 1, 2, 3]
    assert loop_indices(4, 3, "reverse").tolist() == [0, 1, 2]  # shorter audio: cut
    assert loop_indices(3, 9, "reverse", pause=2).tolist() == [0, 1, 2, 2, 2, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        loop_indices(4, 9, "bounce")


# ----------------------------------------------------------------------------- masks
def test_tracked_oval_covers_mouth_not_nose():
    ov = oval_from_landmarks(_landmarks(), TrackConfig())
    m = ov.masks(256)[0].numpy()
    for p in (FACE[2], FACE[3], FACE[6], (0.5, 0.62)):  # mouth corners, upper lip, mouth centre
        assert _at(m, p, 256) > 0.95
    assert _at(m, (0.5, 0.82), 256) > 0.95  # the chin (room to drop)
    assert _at(m, FACE[4], 256) < 0.05  # bottom of the nose stays original
    assert _at(m, FACE[0], 256) < 0.01 and _at(m, (0.5, 0.2), 256) < 0.01  # eyes, forehead


def test_tracked_oval_follows_the_face():
    a = oval_from_landmarks(_landmarks(shift=0.0)).masks(128)[0]
    b = oval_from_landmarks(_landmarks(shift=0.1)).masks(128)[0]
    shifted = torch.roll(a, int(0.1 * 128), dims=1)
    assert (shifted - b).abs()[:, 20:100].mean() < 0.02


def test_bottom_extends_below_chin():
    small = oval_from_landmarks(_landmarks(), TrackConfig(bottom=0.1)).masks(256)[0].numpy()
    big = oval_from_landmarks(_landmarks(), TrackConfig(bottom=0.4)).masks(256)[0].numpy()
    p = (0.5, 0.84)  # a bit below the chin
    assert _at(big, p, 256) > 0.9 and _at(small, p, 256) < _at(big, p, 256)


def test_spread_up_reaches_the_nose_but_not_the_eyes():
    base = oval_from_landmarks(_landmarks(), TrackConfig(spread=0.3)).masks(256)[0].numpy()
    up = oval_from_landmarks(_landmarks(), TrackConfig(spread=0.3, spread_up=1.0)).masks(256)[0].numpy()
    nose = (0.5, 0.47)  # above the bottom of the nose
    assert _at(base, nose, 256) < 0.05  # spread alone fades out at the bottom of the nose
    assert _at(up, nose, 256) > 0.3  # spread_up lets it reach the nose
    assert _at(up, FACE[0], 256) < 0.05 and _at(up, (0.5, 0.40), 256) < 0.05  # eyes stay original


def test_oval_needs_landmarks():
    lm = _landmarks()
    lm[:15] = np.nan
    assert oval_from_landmarks(lm) is None


def test_lower_face_mask():
    m = lower_face_mask(128).numpy()
    assert m[int(0.68 * 128), 64] > 0.99 and m[10, 10] < 1e-3


# ----------------------------------------------------------------------------- skipping the no-audio pass
class _PointwiseModel(torch.nn.Module):
    """Stand-in transformer where each sample only sees itself (so equivalence is exact)."""

    def forward(self, x, zc, t, text, audio):
        a = audio.mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        return torch.tanh(0.5 * x + 0.1 * zc + a + t.view(-1, 1, 1, 1, 1) / 1000)


def _denoise(skip):
    pipe = LipSyncPipeline(_PointwiseModel(), None, None, None, device="cpu", dtype=torch.float32)
    calls = []
    fwd = pipe.transformer.forward
    pipe.transformer.forward = lambda *a: (calls.append(a[0].shape[0]), fwd(*a))[1]
    cfg = InferenceConfig(steps=10, skip_uncond=skip)
    z = torch.randn(1, 4, 5, 8, 8, generator=torch.Generator().manual_seed(1))
    mask = torch.zeros(5, 8, 8)
    mask[:, 3:7, 2:6] = 1
    x = pipe._denoise_window(z, torch.zeros(1, 4, 8), torch.ones(1, 5, 4), torch.zeros(1, 5, 4), mask,
                             torch.tensor([[0.5, 0.6]] * 5), torch.full((5,), 0.55), False, cfg,
                             torch.Generator().manual_seed(0))
    return x, calls, mask


def test_skip_uncond_exact_inside_target():
    a, calls_a, mask = _denoise(False)
    b, calls_b, _ = _denoise(True)
    assert calls_a == [2] * 10 and calls_b.count(1) == 2  # the last two steps need no guidance at omega 4.5
    inside = (mask > 0.5)[None, None].expand_as(a)
    assert (a - b)[inside].abs().max() < 1e-5
    dil = torch.nn.functional.max_pool2d(mask[:, None], 3, 1, 1)[:, 0] > 0.5
    far = (~dil)[None, None].expand_as(a)
    assert (a - b)[far].abs().max() < 1e-6  # only the released rim may differ


def test_cond_only_rule():
    cfg = InferenceConfig(dscfg=DSCFGConfig(omega_peak=4.5, gamma=1.5))
    assert not LipSyncPipeline._cond_only(0.6, cfg, True)
    assert LipSyncPipeline._cond_only(0.36, cfg, True)  # 4.5 * 0.36**1.5 < 1
    assert not LipSyncPipeline._cond_only(0.36, cfg, False)  # no region lock -> keep exact CFG
    cfg.cfg_stop_sigma = 0.5
    assert LipSyncPipeline._cond_only(0.45, cfg, True)


# ----------------------------------------------------------------------------- LoRA merge
def test_merge_lora_same_output(tiny_base, tiny_cfg):
    from wilrosync.models.dit_lipsync import WanLipSyncTransformer
    from wilrosync.models.lora import add_lora

    model = add_lora(WanLipSyncTransformer(tiny_base, tiny_cfg).eval(), rank=4)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.05)
    from wilrosync.models.audio import frame_audio_windows

    torch.manual_seed(0)
    x = torch.randn(1, 4, 3, 8, 8)
    pipe = LipSyncPipeline(model, None, None, None, device="cpu", dtype=torch.float32)
    win = frame_audio_windows(torch.randn(20, 3, 8), 9, 25.0, window=tiny_cfg.audio_window)
    with torch.no_grad():
        tokens = model.encode_audio(win.unsqueeze(0))
    args = (x, x.clone(), torch.tensor([500.0]), torch.randn(1, 12, 16), tokens)
    with torch.no_grad():
        before = model(*args)
        assert pipe.merge_lora() > 0
        after = model(*args)
    assert torch.allclose(before, after, atol=1e-4)


# ----------------------------------------------------------------------------- end to end
class _FakeDetector:
    def detect(self, frame, **kw):
        return np.array([[40.0, 20.0, 88.0, 76.0, 0.99]])


class _FakeLandmarker:
    def __call__(self, crops):
        return _landmarks(len(crops))


def _media(tmp_path, t=12):
    from wilrosync.io.media import write_video, write_wav

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:128]
    base = np.stack([xx * 2, yy * 2, 255 - xx - yy], -1)  # smooth image: survives H.264 almost unchanged
    frames = np.stack([(base + 3 * i) % 256 for i in range(t)]).astype(np.uint8)
    video, audio = str(tmp_path / "in.mp4"), str(tmp_path / "a.wav")
    write_video(video, frames, 25.0)
    write_wav(audio, rng.standard_normal(int(16000 * 18 / 25)).astype(np.float32) * 0.1)  # 18 frames
    return video, audio


def test_preview_needs_no_model(tmp_path, monkeypatch):
    import wilrosync.pipeline.face as face

    monkeypatch.setattr(face, "track_oval", lambda crops, cfg, lm: oval_from_landmarks(_landmarks(len(crops)), cfg))
    video, audio = _media(tmp_path)
    out = lipsync_face(None, video, audio, str(tmp_path / "prev.mp4"), preview=True, size=64,
                       detector=_FakeDetector())
    from wilrosync.io.media import read_video

    assert read_video(out)[0].shape[1:3] == (64, 64)


def test_lipsync_face_end_to_end(tmp_path, tiny_base, tiny_cfg):
    from test_smoke import _Audio, _Text, _tiny_vae

    from wilrosync.io.media import read_video
    from wilrosync.models.dit_lipsync import WanLipSyncTransformer

    model = WanLipSyncTransformer(tiny_base, tiny_cfg).eval()
    pipe = LipSyncPipeline(model, _tiny_vae(), _Text(), _Audio(), device="cpu", dtype=torch.float32)
    video, audio = _media(tmp_path)
    cfg = face_infer_config(steps=2, window=9)
    out = lipsync_face(pipe, video, audio, str(tmp_path / "out.mp4"), cfg=cfg, size=64, detector=_FakeDetector(),
                       landmarker=_FakeLandmarker())
    synced, _ = read_video(out)
    src, _ = read_video(video)
    assert synced.shape == (18, 96, 128, 3)  # follows the audio (ping-pong)
    assert np.abs(synced[:12, :, :20].astype(int) - src[:, :, :20].astype(int)).mean() < 3  # far from the face
