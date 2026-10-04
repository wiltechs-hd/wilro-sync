"""SyncNet scoring: offset sign convention, cropping, filtering and prepare_clips integration."""

import importlib.util
import json
import os

import numpy as np
import pytest
import torch

from wilrosync.eval.syncnet import (
    SyncNet,
    SyncResult,
    crop_faces,
    medfilt,
    resample_indices,
    sync_offset,
)
from wilrosync.models.audio import frame_audio_windows


def test_offset_sign_matches_audio_window_shift():
    """Audio lagging the video by k frames -> offset == -k, and offset_frames = -offset re-aligns it."""
    g = torch.Generator().manual_seed(0)
    base = torch.randn(80, 32, generator=g)
    k = 3
    im = base[10:60]
    cc = base[10 - k: 60 - k]  # audio feature at frame i shows what the lips did at frame i - k -> lagging
    offset, conf, _ = sync_offset(im, cc, vshift=8)
    assert offset == -k and conf > 1
    # the sound of video frame v is at audio frame v + k = v - offset -> offset_frames = -offset
    feats = torch.arange(200).float().view(200, 1, 1)  # 50 Hz feature carrying its own index
    win = frame_audio_windows(feats, num_frames=20, fps=25, window=2, offset_frames=-offset)
    v = 10
    assert win[v, :, 0, 0].tolist() == [2.0 * (v + k) - 1, 2.0 * (v + k)]


def test_offset_zero_and_lead():
    g = torch.Generator().manual_seed(1)
    base = torch.randn(60, 16, generator=g)
    assert sync_offset(base[5:50], base[5:50])[0] == 0
    assert sync_offset(base[5:50], base[7:52])[0] == 2  # audio leads by 2


def test_medfilt_and_resample():
    x = np.array([1, 9, 2, 3, 4], dtype=float)
    assert medfilt(x, 3).tolist() == [1, 2, 3, 3, 3]
    assert resample_indices(30, 30.0, 25.0).tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17,
                                                         18, 19, 20, 22, 23, 24, 25, 26, 28, 29]


def test_crop_faces_geometry():
    frames = np.zeros((3, 200, 300, 3), np.uint8)
    frames[:, 50:150, 100:200] = (255, 0, 0)  # red face box (RGB)
    crops = crop_faces(frames, np.tile([100, 50, 200, 150], (3, 1)), smooth=0)
    assert crops.shape == (3, 3, 224, 224)
    centre = crops[0, :, 60, 112]
    assert centre[2] > 200 and centre[0] < 50  # BGR: red is the last channel


def test_scorer_runs_with_random_weights():
    sn = SyncNet(device="cpu", load_weights=False)
    frames = np.random.default_rng(0).integers(0, 255, (16, 64, 64, 3), dtype=np.uint8)
    wav = np.random.default_rng(1).standard_normal(16000).astype(np.float32) * 0.1
    r = sn.score_video(frames, wav, 25.0, assume_cropped=True, vshift=3)
    assert isinstance(r, SyncResult) and np.isfinite(r.conf) and abs(r.offset) <= 3


class _StubScorer:
    def __init__(self, table):
        self.table, self.calls = table, 0

    def score_file(self, path, **kw):
        self.calls += 1
        v = self.table[os.path.basename(path)]
        return None if v is None else SyncResult(v[0], v[1], 7.0, 100)


def test_filter_manifest_and_resume(tmp_path):
    from wilrosync.data.sync_filter import filter_manifest

    rows = [{"kind": "arbitrary", "video": f"{n}.mp4"} for n in "abcd"]
    rows.append({"kind": "pseudo_pair", "cond": "a.mp4", "target": "e.mp4"})
    man = tmp_path / "m.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in rows))
    table = {"a.mp4": (1, 6.0), "b.mp4": (0, 1.5), "c.mp4": (5, 8.0), "d.mp4": None, "e.mp4": (-2, 4.0)}
    stub = _StubScorer(table)
    out = tmp_path / "f.jsonl"
    c = filter_manifest(str(man), str(tmp_path), str(out), scorer=stub, progress=None)
    assert c == {"total": 5, "kept": 2, "low_conf": 1, "offset": 1, "error": 1}
    kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert [k.get("video", k.get("target")) for k in kept] == ["a.mp4", "e.mp4"]
    assert kept[0]["av_offset"] == 1 and kept[1]["av_offset"] == -2
    stub2 = _StubScorer(table)
    filter_manifest(str(man), str(tmp_path), str(out), scorer=stub2, progress=None, min_conf=1.0)
    assert stub2.calls == 0  # everything reused from the report
    assert len(out.read_text().splitlines()) == 3  # thresholds re-applied without re-scoring


def test_prepare_clips_applies_av_offset(tmp_path):
    from test_prepare_clips import _make_video, _StubEncoders

    class _Enc(_StubEncoders):
        def whisper(self, wav):  # feature value == its own 50 Hz index
            n = int(len(wav) / 16000 * 50) + 1
            return torch.arange(n).float().view(n, 1, 1).expand(n, 3, 8).clone()

    spec = importlib.util.spec_from_file_location("prepare_clips", "scripts/prepare_clips.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_video(str(raw / "a.mp4"), seconds=2)
    for off in (0, 2):
        man = tmp_path / f"m{off}.jsonl"
        man.write_text(json.dumps({"kind": "pseudo_pair", "cond": "a.mp4", "target": "a.mp4", "av_offset": off}))
        out = tmp_path / f"lat{off}"
        mod.main(["--manifest", str(man), "--root", str(raw), "--out", str(out), "--size", "16", "16",
                  "--frames", "9", "--audio-window", "2", "--seed", "0"], encoders=_Enc())
    from safetensors.torch import load_file

    a0 = load_file(str(next((tmp_path / "lat0").glob("*.safetensors"))))["audio"].float()
    a2 = load_file(str(next((tmp_path / "lat2").glob("*.safetensors"))))["audio"].float()
    # offset +2 frames (audio leads) -> windows taken 2 frames = 4 whisper steps earlier
    assert torch.allclose(a0[5:] - a2[5:], torch.full_like(a0[5:], 4.0))


WEIGHTS = os.environ.get("WILROSYNC_CACHE", os.path.expanduser("~/.cache/wilrosync"))


@pytest.mark.skipif(not os.path.isfile(os.path.join(WEIGHTS, "example.avi")),
                    reason="needs SyncNet weights + example.avi in WILROSYNC_CACHE")
def test_reference_example_matches_syncnet_python():
    r = SyncNet(device="cpu").score_file(os.path.join(WEIGHTS, "example.avi"), assume_cropped=True)
    # syncnet_python README: AV offset 3, min dist 5.353, confidence 10.021
    assert r.offset == 3 and abs(r.min_dist - 5.353) < 0.05 and abs(r.conf - 10.021) < 0.1
