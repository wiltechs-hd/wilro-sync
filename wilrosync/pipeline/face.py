"""Face-crop lip sync: crop the target face as in training, sync it, paste it back.

    from wilrosync.pipeline.lipsync import LipSyncPipeline
    from wilrosync.pipeline.face import lipsync_face

    pipe = LipSyncPipeline.from_pretrained("runs/first_model/step_0054800")
    lipsync_face(pipe, "in.mp4", "speech.wav", "out.mp4")              # best settings by default

Steps
1. find the target face (S3FD, every 5th frame; ``bbox`` picks the face in multi-face scenes) and cut one
   fixed square window around it (``scale`` x the face), resized to ``size`` (512, the training layout);
2. ``mode="lower"``: build the mouth mask (tracked oval by default, see ``oval.py``); the model may change only
   the mask, everything else is pinned to the source during denoising (region lock), so no seam.
   ``mode="full"``: the whole crop is generated (mask-free, exactly like training);
3. extend or trim the video to the audio length (``loop``), run the pipeline on the crops;
4. paste the result back with the same per-frame mask and colour matching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..io.media import pingpong_indices, read_audio, read_video, write_video
from .composite import composite
from .lipsync import InferenceConfig, LipSyncPipeline
from .oval import FIXED_DEFAULT, FaceLandmarker, TrackConfig, TrackedOval, lower_face_mask, track_oval
from .target import TargetSignals

FPS = 25.0  # training frame rate
MASK_RES = 256  # mask render resolution (the masks are smooth, resized on use)
LOOPS = ("reverse", "restart", "hold", "trim")


def face_infer_config(**overrides) -> InferenceConfig:
    """Inference settings tuned for face-crop mode (LSE-C sweeps, see docs/inference_face.md)."""
    # window 81 (not the 49-frame training length): 7.24 vs 4.84 LSE-C on an unseen clip, same checkpoint and settings
    cfg = InferenceConfig(steps=10, tau=0.92, window=81, overlap=0, max_side=512, loop_video=False)
    for k, v in overrides.items():
        if k == "dscfg":
            for dk, dv in v.items():
                setattr(cfg.dscfg, dk, dv)
        elif not hasattr(cfg, k):
            raise AttributeError(f"unknown inference setting {k!r}")
        else:
            setattr(cfg, k, v)
    return cfg


def loop_indices(num_frames: int, length: int, loop: str = "reverse", pause: int = 0) -> np.ndarray:
    """Source frame for each output frame when the audio is longer (or shorter) than the video.

    reverse: 0..T-1, T-2..1, 0..  (ping-pong, no jump; ``pause`` holds each end for that many frames)
    restart: 0..T-1, 0..T-1       (jumps back to the start)
    hold:    0..T-1, then the last frame          trim: stop at the end of the video"""
    if loop not in LOOPS:
        raise ValueError(f"loop must be one of {LOOPS}")
    t = np.arange(min(length, num_frames) if loop == "trim" else length)
    if loop == "reverse":
        if pause <= 0 or num_frames == 1 or len(t) <= num_frames:
            return pingpong_indices(num_frames, len(t))
        cycle = np.concatenate([np.arange(num_frames), np.full(pause, num_frames - 1),
                                np.arange(num_frames - 2, 0, -1), np.full(pause, 0)])
        return cycle[t % len(cycle)]
    if loop == "restart":
        return t % num_frames
    return np.minimum(t, num_frames - 1)


@dataclass
class FaceWindow:
    """Fixed crop window in source pixels: square (x, y, side), possibly partly outside the frame."""

    x: int
    y: int
    side: int
    frame_wh: tuple[int, int]

    @property
    def inside(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1): the part of the window inside the frame."""
        w, h = self.frame_wh
        return max(self.x, 0), max(self.y, 0), min(self.x + self.side, w), min(self.y + self.side, h)

    @property
    def offset(self) -> tuple[int, int]:
        x0, y0, _, _ = self.inside
        return x0 - self.x, y0 - self.y


_DETECTOR: dict = {}


def find_face_window(frames: np.ndarray, bbox=None, scale: float = 1.8, detector=None,
                     det_every: int = 5) -> FaceWindow:
    """Track the target face (``bbox`` in the first frame, else the largest face) and return one crop window."""
    from ..data.face_crop import square_window
    from ..eval.syncnet import S3FDDetector, track_face

    det = detector or _DETECTOR.setdefault("s3fd", S3FDDetector())
    boxes = track_face(det, frames[::det_every], init_box=bbox)
    if boxes is None:
        raise RuntimeError("no stable face found; pass bbox=(x0, y0, x1, y1)")
    h, w = frames.shape[1:3]
    face = float(np.median(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])))
    motion = max(np.ptp((boxes[:, 0] + boxes[:, 2]) / 2), np.ptp((boxes[:, 1] + boxes[:, 3]) / 2))
    if motion > 0.6 * face:
        print(f"warning: face moves {motion / face:.1f}x its size; a fixed crop may cut it")
    x, y, side = square_window(boxes, (w, h), scale)
    print(f"face {face:.0f}px -> crop window x={x} y={y} side={side}")
    return FaceWindow(x, y, side, (w, h))


def crop_faces(frames: np.ndarray, win: FaceWindow, size: int, pad_value: float = 110.0) -> np.ndarray:
    """[T, H, W, 3] -> [T, size, size, 3] uint8 crops (grey where the window leaves the frame)."""
    x0, y0, x1, y1 = win.inside
    ox, oy = win.offset
    out = np.empty((len(frames), size, size, 3), np.uint8)
    for s in range(0, len(frames), 32):
        c = torch.full((min(32, len(frames) - s), 3, win.side, win.side), pad_value)
        c[:, :, oy:oy + y1 - y0, ox:ox + x1 - x0] = torch.from_numpy(
            frames[s:s + 32, y0:y1, x0:x1]).permute(0, 3, 1, 2).float()
        c = F.interpolate(c, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        out[s:s + 32] = c.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    return out


def paste_faces(frames: np.ndarray, gen: np.ndarray, win: FaceWindow, masks, feather: float = 0.0,
                color_match: bool = True, chunk: int = 16) -> np.ndarray:
    """Blend generated crops ``gen`` [N, S, S, 3] into ``frames`` [N, H, W, 3] (in place, also returned).

    ``masks(start, end)`` returns the [n, side, side] blend masks for those output frames."""
    x0, y0, x1, y1 = win.inside
    ox, oy = win.offset
    side = win.side
    for s in range(0, len(gen), chunk):
        n = len(gen[s:s + chunk])
        m = masks(s, s + n)[:, oy:oy + y1 - y0, ox:ox + x1 - x0]
        g = torch.from_numpy(gen[s:s + n]).permute(0, 3, 1, 2).float() / 127.5 - 1
        g = F.interpolate(g, size=(side, side), mode="bicubic", align_corners=False).clamp(-1, 1)
        g = g[:, :, oy:oy + y1 - y0, ox:ox + x1 - x0]
        src = torch.from_numpy(frames[s:s + n, y0:y1, x0:x1]).permute(0, 3, 1, 2).float() / 127.5 - 1
        comp = composite(g, src, m, feather_px=feather, color_match=color_match)
        frames[s:s + n, y0:y1, x0:x1] = ((comp + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    return frames


def draw_mask_preview(crops: np.ndarray, masks) -> np.ndarray:
    """Crops with the mask drawn: green = generated, red = soft edge (blend), untouched elsewhere."""
    out = crops.copy()
    size = crops.shape[1]
    for s in range(0, len(crops), 64):
        m = masks(s, min(s + 64, len(crops)), size).numpy()[..., None]
        band = (m > 0.05) & (m < 0.95)
        p = out[s:s + 64].astype(np.float32)
        p = np.where(band, 0.5 * p + [127, 0, 0], np.where(m >= 0.95, p * [0.8, 1.0, 0.8], p))
        out[s:s + 64] = p.clip(0, 255).astype(np.uint8)
    return out


def lipsync_face(pipe: LipSyncPipeline | None, video: str, audio: str | None, out: str, *, bbox=None,
                 cfg: InferenceConfig | None = None, size: int = 512, scale: float = 1.8, mode: str = "lower",
                 loop: str = "reverse", pause: int = 0, oval: str | dict = "track",
                 track: TrackConfig | dict | None = None, preview: bool = False, detector=None,
                 landmarker: FaceLandmarker | None = None) -> str:
    """Lip-sync ``video`` to ``audio`` through a face crop and write ``out``.

    mode:    "lower" (only the mouth mask is generated) or "full" (the whole crop).
    oval:    "track" (follows the mouth, ``track`` sets its shape) or a dict for a fixed ellipse
             (cx, cy, rx, ry, feather; e.g. ``oval.FIXED_OLD``).
    loop:    reverse | restart | hold | trim, when the audio is longer than the video.
    preview: write the crop with the mask drawn instead of syncing (no model needed; ``pipe`` may be None).
    cfg:     inference settings; default ``face_infer_config()`` (10 steps, 81-frame windows, no overlap)."""
    if mode not in ("lower", "full"):
        raise ValueError("mode must be 'lower' or 'full'")
    frames, _ = read_video(video, fps=FPS)
    t_src = len(frames)
    win = find_face_window(frames, bbox, scale, detector)
    crops = crop_faces(frames, win, size)

    tracked: TrackedOval | None = None
    fixed = {**FIXED_DEFAULT, **(oval if isinstance(oval, dict) else {})}
    if mode == "lower" and oval == "track":
        tcfg = track if isinstance(track, TrackConfig) else TrackConfig(**(track or {}))
        tracked = track_oval(crops, tcfg, landmarker)
        if tracked is None:
            print("oval: face landmarks found in < 50% of frames -> fixed oval")
        else:
            print(f"oval: tracked {tcfg.to_dict()}")

    def masks_for(src_idx, res: int) -> torch.Tensor:  # [n, res, res] for these source frames
        if mode == "full":
            b = max(2, int(0.08 * res))
            m = torch.zeros(res, res)
            m[b:-b, b:-b] = 1.0
            return m.expand(len(src_idx), res, res)
        if tracked is not None:
            return tracked.masks(res, src_idx)
        return lower_face_mask(res, **fixed).expand(len(src_idx), res, res)

    if preview:
        prev = draw_mask_preview(crops, lambda s, e, r: F.interpolate(
            masks_for(np.arange(s, e), MASK_RES)[:, None], size=(r, r), mode="bilinear", align_corners=False)[:, 0])
        write_video(out, prev, FPS)
        print(f"mask preview ({t_src} frames) -> {out}")
        return out

    if pipe is None:
        raise ValueError("pipe is required unless preview=True")
    wav = read_audio(audio)
    idx = loop_indices(t_src, int(len(wav) / 16000 * FPS), loop, pause)
    n = len(idx)
    cfg = cfg or face_infer_config()
    cfg.max_side, cfg.loop_video = size, False
    target = None
    if mode == "lower":
        mouth = torch.from_numpy(tracked.mouth[idx]) if tracked is not None \
            else torch.tensor([0.5, 0.6]).expand(n, 2).clone()  # same layout as the training crops
        target = TargetSignals(mask=masks_for(idx, MASK_RES).contiguous(), mouth=mouth.float(),
                               face_width=torch.full((n,), 0.55), presence=torch.ones(n, dtype=torch.bool))
    res = pipe(crops[idx], FPS, wav, target=target, cfg=cfg)  # uint8 [N, size, size, 3]

    feather = max(2, int(0.08 * win.side)) if mode == "full" else 0.0  # the lower masks are already soft

    def paste_masks(s: int, e: int) -> torch.Tensor:
        m = masks_for(idx[s:e], MASK_RES)
        return F.interpolate(m[:, None], size=(win.side, win.side), mode="bilinear", align_corners=False)[:, 0]

    result = paste_faces(frames[idx[:len(res)]].copy(), res, win, paste_masks, feather)
    write_video(out, result, FPS, audio_path=audio)
    print(f"{len(result)} frames -> {out}")
    return out
