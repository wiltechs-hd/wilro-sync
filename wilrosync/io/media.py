"""Video / audio I/O through the ffmpeg binary bundled with ``imageio-ffmpeg``."""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

AUDIO_SR = 16000


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def read_video(path: str, max_frames: int | None = None, fps: float | None = None) -> tuple[np.ndarray, float]:
    """Returns (frames uint8 [T, H, W, 3], fps). ``fps`` resamples the video (e.g. 25 for training)."""
    import imageio_ffmpeg

    params = ["-r", str(fps)] if fps else None
    gen = imageio_ffmpeg.read_frames(path, pix_fmt="rgb24", output_params=params)
    meta = next(gen)
    w, h = meta["size"]
    fps = float(fps or meta.get("fps") or 25.0)
    frames = []
    for i, buf in enumerate(gen):
        if max_frames is not None and i >= max_frames:
            gen.close()
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3))
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames), fps


def write_video(path: str, frames: np.ndarray, fps: float, audio_path: str | None = None, crf: int = 17) -> None:
    """Write uint8 frames [T, H, W, 3] as H.264 (+ optional audio track, trimmed to the video)."""
    t, h, w, _ = frames.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        silent = os.path.join(tmp, "video.mp4") if audio_path else path
        cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", silent]
        proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode())
        if audio_path:
            cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", silent, "-i", audio_path, "-map", "0:v:0",
                   "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", path]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode:
                raise RuntimeError(proc.stderr.decode())


def read_audio(path: str, sr: int = AUDIO_SR) -> np.ndarray:
    """Decode any audio/video file to mono float32 at ``sr`` Hz."""
    cmd = [_ffmpeg(), "-loglevel", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
           "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode())
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def write_wav(path: str, wav: np.ndarray, sr: int = AUDIO_SR) -> None:
    import wave

    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def pingpong_indices(num_frames: int, length: int) -> np.ndarray:
    """Frame indices that extend a clip to ``length`` by playing it forward, backward, forward...
    (no jump cut). Truncates when ``length <= num_frames``."""
    if length <= num_frames:
        return np.arange(length)
    if num_frames == 1:
        return np.zeros(length, dtype=np.int64)
    cycle = np.concatenate([np.arange(num_frames), np.arange(num_frames - 2, 0, -1)])
    return cycle[np.arange(length) % len(cycle)]
