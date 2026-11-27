"""Lip-sync a whole episode with several characters (``wilro-sync episode``).

Each character comes with a reference photo of their face and their own voice track (same length as the episode,
silence where they do not speak, e.g. one text-to-speech track per character). The episode is

1. split into shots at the cuts (PySceneDetect), so the face track, the reference frames and the generation
   windows never cross a cut;
2. per shot, every face is matched to the characters by face identity (OpenCV YuNet + SFace embeddings): each face
   to at most one character, faces of nobody in the list (extras) are left alone, faces smaller than ``min_face``
   pixels are skipped (lips too small to sync);
3. per shot, every matched character who speaks in that shot is lip-synced with their own track (one pass per
   face, each pass changes only that mouth); a matched character who is silent is left as filmed
   (``silent="keep"``) or synced to silence so the mouth stays closed (``silent="close"``);
4. the shots are joined again and the full soundtrack (``audio``, or the sum of the character tracks) is laid over
   the result.

``lipsync_episode`` returns a per-shot report (who was found where, with what similarity, what was synced).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
from dataclasses import asdict, dataclass, field

import numpy as np

from ..io.media import AUDIO_SR, _ffmpeg, probe_video, read_audio, write_wav

FPS = 25
SPF = AUDIO_SR // FPS  # audio samples per video frame
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
             "face_recognition_sface_2021dec.onnx")


@dataclass
class Character:
    name: str
    face: str   # reference photo (the character's face clearly visible)
    audio: str  # their voice track, as long as the episode, silence where they do not speak


def parse_character(spec: str) -> Character:
    """``name=face.jpg:voice.wav`` -> Character."""
    try:
        name, rest = spec.split("=", 1)
        face, audio = rest.rsplit(":", 1)
    except ValueError:
        raise ValueError(f"character must be name=face.jpg:voice.wav, got {spec!r}") from None
    if not name or not face or not audio:
        raise ValueError(f"character must be name=face.jpg:voice.wav, got {spec!r}")
    return Character(name.strip(), face.strip(), audio.strip())


@dataclass
class ShotReport:
    index: int
    start: int  # first frame (at 25 fps)
    end: int    # one past the last frame
    faces: int = 0
    matched: dict = field(default_factory=dict)   # character -> {"bbox", "similarity", "action"}
    note: str = ""


# ----------------------------------------------------------------------------- shots
def detect_shots(video: str, num_frames: int, threshold: float = 27.0, min_len: int = 15) -> list[tuple[int, int]]:
    """Frame ranges [start, end) at 25 fps covering the whole video, split at cuts."""
    try:
        from scenedetect import ContentDetector, detect
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("episode mode needs PySceneDetect: pip install 'wilro-sync[episode]'") from e
    scenes = detect(video, ContentDetector(threshold=threshold, min_scene_len=min_len))
    cuts = sorted({int(round(s.get_seconds() * FPS)) for s, _ in scenes[1:]})
    return _ranges([c for c in cuts if 0 < c < num_frames], num_frames)


def _ranges(cuts: list[int], n: int) -> list[tuple[int, int]]:
    bounds = [0, *cuts, n]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


# ----------------------------------------------------------------------------- faces
def _download(url: str, path: str) -> str:
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


class FaceIdentifier:
    """OpenCV YuNet (detection + landmarks) and SFace (identity embedding); models downloaded once into the weights
    cache. ``faces(image)`` -> [(box x0,y0,x1,y1, unit embedding)]."""

    threshold = 0.363  # SFace's recommended cosine threshold for "same person"

    def __init__(self, model_dir: str | None = None) -> None:
        import cv2

        if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
            raise ImportError("episode mode needs OpenCV >= 4.8 with FaceDetectorYN / FaceRecognizerSF")
        from ..eval.syncnet import weights_dir

        d = model_dir or weights_dir()
        self._cv2 = cv2
        self._det = cv2.FaceDetectorYN.create(_download(YUNET_URL, os.path.join(d, "yunet_2023mar.onnx")), "",
                                              (320, 320), 0.8, 0.3, 5000)
        self._rec = cv2.FaceRecognizerSF.create(_download(SFACE_URL, os.path.join(d, "sface_2021dec.onnx")), "")

    def faces(self, image: np.ndarray) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
        """RGB uint8 [H, W, 3] -> faces with their identity embeddings."""
        bgr = np.ascontiguousarray(image[..., ::-1])
        h, w = bgr.shape[:2]
        self._det.setInputSize((w, h))
        _, dets = self._det.detect(bgr)
        out = []
        for d in (dets if dets is not None else []):
            emb = self._rec.feature(self._rec.alignCrop(bgr, d)).reshape(-1).astype(np.float64)
            x, y, bw, bh = (float(v) for v in d[:4])
            out.append(((x, y, x + bw, y + bh), emb / max(np.linalg.norm(emb), 1e-9)))
        return out

    def reference(self, path: str) -> np.ndarray:
        """Embedding of the largest face in a reference photo."""
        import imageio.v3 as iio

        img = np.asarray(iio.imread(path))[..., :3]
        faces = self.faces(img)
        if not faces:
            raise ValueError(f"no face found in the reference photo {path}")
        return max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))[1]


def match_faces(faces: list, refs: dict[str, np.ndarray], threshold: float, min_face: int = 0) -> dict:
    """Greedy best-first assignment: each face to at most one character and each character to at most one face,
    above ``threshold`` cosine similarity. -> {name: (box, similarity)}."""
    pairs = []
    for i, (box, emb) in enumerate(faces):
        if max(box[2] - box[0], box[3] - box[1]) < min_face:
            continue
        for name, ref in refs.items():
            pairs.append((float(np.dot(emb, ref)), i, name))
    out, used = {}, set()
    for sim, i, name in sorted(pairs, reverse=True):
        if sim < threshold or i in used or name in out:
            continue
        out[name] = (faces[i][0], sim)
        used.add(i)
    return out


# ----------------------------------------------------------------------------- audio
def speech_activity(wav: np.ndarray, level: float) -> np.ndarray:
    """Per video frame: True where the track is louder than ``level`` (RMS)."""
    n = len(wav) // SPF
    rms = np.sqrt((wav[: n * SPF].reshape(n, SPF).astype(np.float64) ** 2).mean(1))
    return rms > level


def _fit(wav: np.ndarray, n_samples: int) -> np.ndarray:
    return wav[:n_samples] if len(wav) >= n_samples else np.pad(wav, (0, n_samples - len(wav)))


# ----------------------------------------------------------------------------- video helpers
def _run(*args: str) -> None:
    proc = subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", *args], capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode()[-2000:])


def count_frames(video: str) -> int:
    """Frames at 25 fps (the rate everything here works at)."""
    return max(1, int(round(probe_video(video)["duration"] * FPS)))


def extract_shot(video: str, start: int, end: int, out: str) -> str:
    """Frames [start, end) at 25 fps as their own video (no audio, near-lossless).

    Trims on the input side by time and forces CFR on output: a ``trim``/``select`` filter chain
    rounds the last frame's duration away and each shot would come out one frame short."""
    _run("-ss", f"{start / FPS:.6f}", "-i", video, "-t", f"{(end - start) / FPS:.6f}",
         "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-fps_mode", "cfr", out)
    return out


def grab_frame(video: str, index: int) -> np.ndarray:
    """One frame (RGB uint8) of a 25 fps video."""
    info = probe_video(video)
    w, h = info["width"], info["height"]
    proc = subprocess.run([_ffmpeg(), "-loglevel", "error", "-i", video, "-vf",
                           f"fps={FPS},select=eq(n\\,{index})", "-frames:v", "1", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-"], capture_output=True)
    if proc.returncode or len(proc.stdout) < w * h * 3:
        raise RuntimeError(f"could not read frame {index} of {video}")
    return np.frombuffer(proc.stdout[: w * h * 3], np.uint8).reshape(h, w, 3)


def join_shots(parts: list[str], audio: str, out: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        lst = os.path.join(tmp, "parts.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
        _run("-f", "concat", "-safe", "0", "-i", lst, "-i", audio, "-map", "0:v", "-map", "1:a",
             "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out)
    return out


# ----------------------------------------------------------------------------- the episode
def lipsync_episode(pipe, video: str, characters: list[Character], out: str, audio: str | None = None, *,
                    silent: str = "keep", min_face: int = 80, match_threshold: float | None = None,
                    min_shot: int = 10, speech_min_frames: int = 8, shot_threshold: float = 27.0,
                    identifier=None, shot_detector=None, sync_fn=None, workdir: str | None = None,
                    report: str | None = None, **face_kwargs) -> list[ShotReport]:
    """Lip-sync every speaking character in every shot of ``video`` (see the module docstring).

    silent:          "keep" (leave a silent character as filmed) or "close" (sync them to silence).
    min_face:        faces smaller than this (pixels) are not synced.
    match_threshold: cosine similarity needed to accept a face as a character (default: SFace's 0.363).
    min_shot:        shots shorter than this many frames are left as they are.
    face_kwargs:     passed to ``lipsync_face`` (cfg, mode, oval, reference, ...).
    identifier / shot_detector / sync_fn: replaceable parts (tests, other face models)."""
    from .face import lipsync_face

    if silent not in ("keep", "close"):
        raise ValueError("silent must be 'keep' or 'close'")
    names = [c.name for c in characters]
    if len(set(names)) != len(names) or not names:
        raise ValueError("give each character a unique name")
    ident = identifier or FaceIdentifier()
    threshold = ident.threshold if match_threshold is None else match_threshold
    refs = {c.name: ident.reference(c.face) for c in characters}

    n = count_frames(video)
    shots = (shot_detector or (lambda v, k: detect_shots(v, k, shot_threshold)))(video, n)
    tracks = {c.name: _fit(read_audio(c.audio), n * SPF) for c in characters}
    activity = {}
    for name, wav in tracks.items():  # "speaking" = louder than 15% of the character's own loud level
        frame_rms = np.sqrt((wav.reshape(n, SPF).astype(np.float64) ** 2).mean(1))
        loud = float(np.percentile(frame_rms, 95)) if frame_rms.any() else 0.0
        activity[name] = speech_activity(wav, max(0.15 * loud, 1e-4))
    sync = sync_fn or (lambda v, a, o, bbox: lipsync_face(pipe, v, a, o, bbox=bbox, loop="trim", **face_kwargs))

    reports: list[ShotReport] = []
    with tempfile.TemporaryDirectory(prefix="wilrosync-episode-", dir=workdir) as tmp:
        parts = []
        for k, (a, b) in enumerate(shots):
            rep = ShotReport(k, a, b)
            reports.append(rep)
            current = extract_shot(video, a, b, os.path.join(tmp, f"shot{k:04d}.mp4"))
            if b - a < min_shot:
                rep.note = f"shorter than {min_shot} frames: left as is"
                parts.append(current)
                continue
            faces = ident.faces(grab_frame(current, min(2, b - a - 1)))
            rep.faces = len(faces)
            matched = match_faces(faces, refs, threshold, min_face)
            for name, (box, sim) in sorted(matched.items(), key=lambda m: -m[1][1]):
                speaks = int(activity[name][a:b].sum()) >= speech_min_frames
                action = "sync" if speaks else ("close" if silent == "close" else "keep")
                rep.matched[name] = {"bbox": [round(v) for v in box], "similarity": round(sim, 3), "action": action}
                if action == "keep":
                    continue
                seg = tracks[name][a * SPF: b * SPF] if speaks else np.zeros((b - a) * SPF, np.float32)
                wav = os.path.join(tmp, f"shot{k:04d}_{name}.wav")
                write_wav(wav, seg)
                nxt = os.path.join(tmp, f"shot{k:04d}_{name}.mp4")
                sync(current, wav, nxt, tuple(float(v) for v in box))
                current = nxt
            print(f"shot {k} [{a}:{b}] faces {rep.faces}: "
                  + (", ".join(f"{m} {v['action']} ({v['similarity']})" for m, v in rep.matched.items()) or "none"))
            parts.append(current)

        if audio:
            mix = audio
        else:  # the sum of the character tracks
            mix = os.path.join(tmp, "mix.wav")
            write_wav(mix, np.clip(sum(tracks.values()), -1.0, 1.0).astype(np.float32))
        join_shots(parts, mix, out)
    if report:
        with open(report, "w") as f:
            json.dump([asdict(r) for r in reports], f, indent=2)
    print(f"{len(shots)} shots, {sum(len(r.matched) for r in reports)} character appearances -> {out}")
    return reports
