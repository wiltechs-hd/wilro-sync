"""prepare_clips.py end to end on a generated video, with stub encoders (no downloads)."""

import importlib.util
import json
import os
import subprocess

import imageio_ffmpeg
import torch
from safetensors.torch import load_file

from wilrosync.data.datasets import LatentClipDataset, TimestepDependentSampler, collate


def _make_video(path, seconds=3):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc=size=96x64:rate=30:duration={seconds}",
           "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path]
    subprocess.run(cmd, check=True)


class _StubEncoders:
    def text(self, caption):
        return torch.randn(7, 16).to(torch.bfloat16)

    def latent(self, clip):  # clip [3, T, h, w] -> [4, F, h/8, w/8]
        t, h, w = clip.shape[1:]
        return torch.randn(4, (t - 1) // 4 + 1, h // 8, w // 8).to(torch.bfloat16)

    def whisper(self, wav):
        return torch.randn(int(len(wav) / 16000 * 50) + 1, 3, 8)


def test_prepare_clips_and_load(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_clips", "scripts/prepare_clips.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_video(str(raw / "a.mp4"))
    _make_video(str(raw / "b.mp4"))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "arbitrary", "video": "a.mp4"},
        {"kind": "pseudo_pair", "cond": "a.mp4", "target": "b.mp4"},
    ]))
    out = tmp_path / "latents" / "set"
    n = mod.main(["--manifest", str(manifest), "--root", str(raw), "--out", str(out), "--size", "32", "48",
                  "--frames", "9", "--clips-per-video", "2", "--audio-window", "2"], encoders=_StubEncoders())
    assert n == 3
    assert (tmp_path / "latents" / "null_text.safetensors").exists()
    d = load_file(str(next(out.glob("arbitrary_*.safetensors"))))
    assert d["z_ab"].shape == (4, 3, 4, 6) and d["audio"].shape == (9, 2, 3, 8)
    index = str(out / "index.jsonl")
    pairs = LatentClipDataset.from_index(index, kind="pseudo_pair")
    arb = LatentClipDataset.from_index(index, kind="arbitrary")
    assert len(pairs) == 1 and len(arb) == 2 and os.path.isabs(arb.rows[0]["path"])
    sampler = iter(TimestepDependentSampler(pairs, arb, threshold=0.85))
    batch = collate([next(sampler) for _ in range(4)], text_max_len=32)
    assert batch["text"].shape == (4, 32, 16) and batch["z_cd"].shape == (4, 4, 3, 4, 6)


def test_video_io_roundtrip(tmp_path):
    import numpy as np

    from wilrosync.io.media import read_audio, read_video, write_video, write_wav

    frames = (np.random.default_rng(0).random((10, 32, 48, 3)) * 255).astype(np.uint8)
    wav_path = str(tmp_path / "a.wav")
    write_wav(wav_path, np.sin(np.linspace(0, 400, 16000)).astype(np.float32) * 0.3)
    out = str(tmp_path / "o.mp4")
    write_video(out, frames, 25.0, audio_path=wav_path)
    back, fps = read_video(out)
    assert back.shape == frames.shape and abs(fps - 25.0) < 1e-3
    assert len(read_audio(out)) > 0
