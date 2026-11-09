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


def test_face_defaults_match_the_tuned_settings():
    cfg = face_infer_config()
    assert (cfg.steps, cfg.window, cfg.overlap, cfg.tau, cfg.max_side) == (10, 129, 0, 0.92, 512)
    assert face_infer_config(window=49, dscfg={"omega_peak": 3.5}).dscfg.omega_peak == 3.5


# ----------------------------------------------------------------------------- moving crop window
class _SquareDetector:
    """Finds the white square drawn into the synthetic frames (the 'face')."""

    def detect(self, frame, **kw):
        ys, xs = np.nonzero(frame[..., 0] > 200)
        if len(xs) == 0:
            return np.zeros((0, 5))
        return np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.99]], float)


def _moving_face_frames(t=24, h=120, w=200, face=30):
    frames = np.full((t, h, w, 3), 40, np.uint8)
    xs = (20 + np.arange(t) * 5).astype(int)  # moves 5 px per frame: 115 px = ~4x its size
    for i, x in enumerate(xs):
        frames[i, 40:40 + face, x:x + face] = 255
    return frames, xs


def test_tracked_crop_keeps_the_face_centred():
    from wilrosync.pipeline.face import crop_faces, find_face_window

    frames, _ = _moving_face_frames()
    fixed = find_face_window(frames, None, 1.8, _SquareDetector(), crop="fixed")
    moving = find_face_window(frames, None, 1.8, _SquareDetector(), crop="track", smooth=3)
    assert not fixed.moving and moving.moving and moving.side < fixed.side  # no room for motion needed

    def centre_x(crops):
        return np.array([np.nonzero(c[..., 0] > 200)[1].mean() / c.shape[1] for c in crops])

    cf, ct = centre_x(crop_faces(frames, fixed, 64)), centre_x(crop_faces(frames, moving, 64))
    assert np.ptp(cf) > 0.4  # the face slides across the fixed crop
    # stays near the centre of the tracked crop (the lazy camera trails by at most the dead zone: 0.15 / 1.8)
    assert np.ptp(ct[3:-3]) < 0.05 and abs(ct[3:-3].mean() - 0.5) < 0.1


def test_tracked_crop_paste_back_round_trip():
    from wilrosync.pipeline.face import crop_faces, find_face_window, paste_faces

    frames, _ = _moving_face_frames()
    win = find_face_window(frames, None, 1.8, _SquareDetector(), crop="track", smooth=3)
    crops = crop_faces(frames, win, win.side)  # no resize: the round trip should be exact
    ones = lambda s, e: torch.ones(e - s, win.side, win.side)  # noqa: E731
    out = paste_faces(frames.copy(), crops, win, ones, color_match=False)
    assert np.abs(out.astype(int) - frames.astype(int)).max() <= 1
    # pasting a black crop only changes pixels inside each frame's own window
    out = paste_faces(frames.copy(), np.zeros_like(crops), win, ones, color_match=False)
    for t in (0, 12, 23):
        x0, y0, x1, y1, _, _ = win.box(t)
        assert out[t, y0:y1, x0:x1].max() == 0
        rest = out[t].copy()
        rest[y0:y1, x0:x1] = frames[t, y0:y1, x0:x1]
        assert np.array_equal(rest, frames[t])


def test_tracked_crop_paste_follows_the_source_frame():
    from wilrosync.pipeline.face import FaceWindow, paste_faces

    frames = np.zeros((3, 40, 60, 3), np.uint8)
    win = FaceWindow([0, 20, 40], [0, 10, 20], 20, (60, 40))
    gen = np.full((2, 20, 20, 3), 255, np.uint8)
    ones = lambda s, e: torch.ones(e - s, 20, 20)  # noqa: E731
    out = paste_faces(frames[[2, 0]].copy(), gen, win, ones, color_match=False, src_idx=[2, 0])
    assert out[0, 20:40, 40:60].min() == 255 and out[0, :20, :40].max() == 0  # output 0 = source frame 2
    assert out[1, 0:20, 0:20].min() == 255 and out[1, 20:, 20:].max() == 0


def test_track_face_follows_fast_motion():
    from wilrosync.eval.syncnet import track_face

    frames, xs = _moving_face_frames()
    boxes = track_face(_SquareDetector(), frames[::5])  # 25 px between detections on a 30 px face
    assert boxes is not None and np.allclose(boxes[:, 0], xs[::5], atol=1)


def test_lipsync_face_tracked_crop_end_to_end(tmp_path, tiny_base, tiny_cfg):
    from test_smoke import _Audio, _Text, _tiny_vae

    from wilrosync.io.media import read_video, write_video, write_wav
    from wilrosync.models.dit_lipsync import WanLipSyncTransformer

    frames, _ = _moving_face_frames(t=12, h=96, w=160)
    video, audio = str(tmp_path / "move.mp4"), str(tmp_path / "a.wav")
    write_video(video, frames, 25.0)
    write_wav(audio, np.zeros(int(16000 * 12 / 25), np.float32))
    model = WanLipSyncTransformer(tiny_base, tiny_cfg).eval()
    pipe = LipSyncPipeline(model, _tiny_vae(), _Text(), _Audio(), device="cpu", dtype=torch.float32)
    out = lipsync_face(pipe, video, audio, str(tmp_path / "out.mp4"), cfg=face_infer_config(steps=2, window=9),
                       size=64, detector=_SquareDetector(), landmarker=_FakeLandmarker(), crop="track")
    synced, _ = read_video(out)
    assert synced.shape == (12, 96, 160, 3)


def test_lazy_follow_ignores_jitter_and_follows_motion():
    from wilrosync.pipeline.face import lazy_follow

    rng = np.random.default_rng(0)
    still = 100 + rng.normal(0, 2, 200)  # detector jitter of ~2 px on a 100 px face, dead zone 15 px
    assert np.ptp(lazy_follow(still, 15.0)) == 0
    moving = np.linspace(100, 400, 200)
    cam = lazy_follow(moving, 15.0)
    assert np.abs(cam - moving).max() <= 15.0 + 1e-6 and cam[-1] > 380


def test_auto_crop_picks_by_motion():
    from wilrosync.pipeline.face import find_face_window

    fast, _ = _moving_face_frames()  # moves ~4x its size
    assert find_face_window(fast, None, 1.8, _SquareDetector(), crop="auto").moving
    still = np.repeat(fast[:1], 24, 0)
    assert not find_face_window(still, None, 1.8, _SquareDetector(), crop="auto").moving


def test_preroll_generates_extra_frames_and_drops_them(tmp_path):
    from wilrosync.io.media import read_video

    calls = {}

    class _EchoPipe:  # returns the crops it was given, records what it got
        def __call__(self, frames, fps, wav, target=None, cfg=None):
            calls.update(n=len(frames), wav=len(wav), first=frames[0].copy(), mask_n=target.mask.shape[0])
            return frames

    video, audio = _media(tmp_path)  # 12 video frames, 18 frames of audio
    out = lipsync_face(_EchoPipe(), video, audio, str(tmp_path / "pre.mp4"), cfg=face_infer_config(steps=1, window=9),
                       size=64, detector=_FakeDetector(), landmarker=_FakeLandmarker(), preroll=4)
    assert calls["n"] == 18 + 4 and calls["mask_n"] == 18 + 4  # 4 extra frames generated ...
    assert calls["wav"] == int(16000 * 18 / 25) + 4 * 640  # ... with 4 frames of silence in front
    assert read_video(out)[0].shape[0] == 18  # ... and dropped: same length as the audio


def test_blend_single_and_raw_out(tmp_path):
    from wilrosync.io.media import read_video

    seen = []

    class _BrightPipe:  # returns white crops, records whether the pipeline was asked to blend
        def __call__(self, frames, fps, wav, target=None, cfg=None):
            seen.append(cfg.composite)
            return np.full_like(frames, 255)

    video, audio = _media(tmp_path)
    kw = dict(size=64, detector=_FakeDetector(), landmarker=_FakeLandmarker())
    outs = {}
    for b in ("double", "single"):
        outs[b] = lipsync_face(_BrightPipe(), video, audio, str(tmp_path / f"{b}.mp4"), blend=b,
                               cfg=face_infer_config(steps=1, window=9), raw_out=str(tmp_path / f"raw_{b}.mp4"), **kw)
    assert seen == [False, False]  # lower mode: the pipeline returns raw crops
    raw = read_video(str(tmp_path / "raw_single.mp4"))[0]
    assert raw.shape[2] == 2 * raw.shape[1]  # source | generated side by side
    single, double = (read_video(outs[b])[0].astype(float) for b in ("single", "double"))
    # single blend: the generated (white) mouth weighs mask, double about mask x feathered mask -> brighter
    assert single.mean() > double.mean()


def test_face_config_chain_default():
    assert face_infer_config().chain == 0 and face_infer_config().window == 129
    assert face_infer_config(chain=5).chain == 5


def test_same_noise_repeats_the_window_noise():
    from wilrosync.pipeline.lipsync import InferenceConfig

    seen = []

    class _Stop(Exception):
        pass

    class _Pipe(LipSyncPipeline):
        def __init__(self):
            self.device, self.dtype = torch.device("cpu"), torch.float32

    pipe = _Pipe()
    import wilrosync.pipeline.lipsync as lp
    orig = lp.add_noise

    def spy(z, noise, sigma):
        seen.append(noise.clone())
        raise _Stop

    lp.add_noise = spy
    try:
        gen = torch.Generator().manual_seed(0)
        for same in (True, True, False, False):
            cfg = InferenceConfig(steps=2, same_noise=same)
            try:
                pipe._denoise_window(torch.zeros(1, 4, 3, 2, 2), None, None, None, None, None, None, True, cfg, gen)
            except _Stop:
                pass
    finally:
        lp.add_noise = orig
    assert torch.equal(seen[0], seen[1])  # same noise in every window
    assert not torch.equal(seen[2], seen[3])  # default: fresh noise per window


def test_postroll_generates_extra_frames_after_the_end_and_drops_them(tmp_path):
    from wilrosync.io.media import read_video

    calls = {}

    class _EchoPipe:
        def __call__(self, frames, fps, wav, target=None, cfg=None):
            calls.update(n=len(frames), wav=len(wav), last=frames[-1].copy(), mask_n=target.mask.shape[0])
            return frames

    video, audio = _media(tmp_path)  # 12 video frames, 18 frames of audio
    out = lipsync_face(_EchoPipe(), video, audio, str(tmp_path / "post.mp4"), cfg=face_infer_config(steps=1, window=9),
                       size=64, detector=_FakeDetector(), landmarker=_FakeLandmarker(), preroll=2, postroll=5)
    assert calls["n"] == 2 + 18 + 5 and calls["mask_n"] == 2 + 18 + 5
    assert calls["wav"] == (2 + 18 + 5) * 640  # silence before and after
    assert read_video(out)[0].shape[0] == 18  # extra frames dropped: same length as the audio
