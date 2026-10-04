import numpy as np
import torch

from wilrosync.io.media import pingpong_indices
from wilrosync.models.audio import AudioProjector, frame_audio_windows
from wilrosync.pipeline.composite import composite
from wilrosync.pipeline.planner import blend_weights, plan_windows, valid_length
from wilrosync.pipeline.target import TargetSignals, latent_frame_groups


def test_latent_groups():
    g = latent_frame_groups(9)
    assert g == [[0], [1, 2, 3, 4], [5, 6, 7, 8]]


def test_bbox_target_and_latent():
    t = TargetSignals.from_bbox(9, 64, 128, (10, 10, 40, 50))
    assert t.mask.shape[0] == 9 and t.mask[0, 30, 25] > 0.9 and t.mask[0, 5, 120] == 0
    m, mouth, fw = t.to_latent((8, 16))
    assert m.shape == (3, 8, 16) and mouth.shape == (3, 2)
    assert m[:, :, 12:].max() == 0  # right side (second face) is never editable
    assert abs(mouth[0, 0].item() - 25 / 128) < 1e-6


def test_planner_cover_and_blend():
    assert valid_length(10) == 13 and valid_length(9) == 9
    n = 200
    wins = plan_windows(n, window=81, overlap=12)
    covered = torch.zeros(n)
    for w in wins:
        assert (w.length - 1) % 4 == 0
        covered[w.start : w.end] += 1
    assert covered.min() >= 1
    weights = blend_weights(wins, n, 12)
    total = torch.zeros(n)
    for w, wt in zip(wins, weights):
        total[w.start : w.end] += wt
    torch.testing.assert_close(total, torch.ones(n))


def test_planner_presence_segments():
    pres = torch.zeros(50, dtype=torch.bool)
    pres[5:15] = True
    pres[30:31] = True  # single frame -> dropped (min_segment=2)
    wins = plan_windows(50, window=81, overlap=12, presence=pres)
    assert len(wins) == 1 and wins[0].start == 5 and wins[0].end == 15 and wins[0].length == 13


def test_audio_windows_and_projector():
    feats = torch.arange(100, dtype=torch.float32).view(100, 1, 1).expand(100, 2, 3)  # 2 s at 50 Hz
    win = frame_audio_windows(feats, num_frames=9, fps=25, window=4)
    assert win.shape == (9, 4, 2, 3)
    assert win[4, :, 0, 0].tolist() == [6.0, 7.0, 8.0, 9.0]  # frame 4 -> t = 0.16 s -> index 8
    proj = AudioProjector(num_layers=2, in_dim=3, dim=8, window=4)
    tok = proj(win.unsqueeze(0))
    assert tok.shape == (1, 3, 16, 8)
    dropped = proj(win.unsqueeze(0), drop=torch.tensor([True]))
    torch.testing.assert_close(dropped[0], proj.null(1, 3)[0])


def test_composite_bit_exact_outside():
    g = torch.Generator().manual_seed(0)
    gen, src = torch.rand(2, 3, 32, 32, generator=g) * 2 - 1, torch.rand(2, 3, 32, 32, generator=g) * 2 - 1
    mask = torch.zeros(2, 32, 32)
    mask[:, 8:24, 8:24] = 1
    out = composite(gen, src, mask, feather_px=4)
    assert torch.equal(out[..., :8, :], src[..., :8, :])
    assert not torch.equal(out[..., 16, 16], src[..., 16, 16])  # inside: generated content


def test_pingpong():
    assert pingpong_indices(4, 9).tolist() == [0, 1, 2, 3, 2, 1, 0, 1, 2]
    assert pingpong_indices(5, 3).tolist() == [0, 1, 2]
    assert np.all(pingpong_indices(1, 3) == 0)
