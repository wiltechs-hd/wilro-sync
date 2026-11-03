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
    """Square crop window in source pixels, per source frame: top-left ``x[t], y[t]`` and a constant ``side``.

    ``crop="fixed"`` uses the same position for every frame; ``crop="track"`` follows the face. The window may
    extend past the frame (that part is grey in the crop and left untouched when pasting back)."""

    x: np.ndarray
    y: np.ndarray
    side: int
    frame_wh: tuple[int, int]

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.int64).reshape(-1)
        self.y = np.asarray(self.y, dtype=np.int64).reshape(-1)

    @property
    def moving(self) -> bool:
        return len(self.x) > 1 and (np.ptp(self.x) > 0 or np.ptp(self.y) > 0)

    def box(self, t: int) -> tuple[int, int, int, int, int, int]:
        """(x0, y0, x1, y1, ox, oy) for source frame ``t``: the part inside the frame and its offset in the window."""
        x, y = int(self.x[t % len(self.x)]), int(self.y[t % len(self.y)])
        w, h = self.frame_wh
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + self.side, w), min(y + self.side, h)
        return x0, y0, x1, y1, x0 - x, y0 - y


_DETECTOR: dict = {}


def smooth_track(values: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average over ``k`` frames (edges padded), no lag."""
    if k <= 1 or len(values) < 2:
        return values.astype(np.float64)
    pad = np.pad(values.astype(np.float64), (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, "valid")


def lazy_follow(centre: np.ndarray, deadzone: float, settle: int = 25) -> np.ndarray:
    """A 'lazy camera' along one axis: it stays still while the face is within ``deadzone`` of it and only moves
    (just enough) when the face leaves that zone. Small head motion and detector jitter never move the crop."""
    c = np.asarray(centre, dtype=np.float64)
    if len(c) == 0 or deadzone <= 0:
        return c.copy()
    p = np.empty_like(c)
    cam = float(np.median(c[:settle]))
    for i, v in enumerate(c):
        d = v - cam
        if abs(d) > deadzone:
            cam += d - np.sign(d) * deadzone
        p[i] = cam
    return p


def find_face_window(frames: np.ndarray, bbox=None, scale: float = 1.8, detector=None, det_every: int | None = None,
                     crop: str = "fixed", smooth: int = 9, y_shift: float = 0.1, deadzone: float = 0.15,
                     auto_motion: float = 0.5) -> FaceWindow:
    """Track the target face (``bbox`` in the first frame, else the largest face) and return the crop window.

    crop="fixed": one square for the whole clip, covering all head motion (detection every 5th frame).
    crop="track": the square follows the face (detection every frame), constant size ``scale`` x face. It moves like
                  a lazy camera: still while the face stays within ``deadzone`` x face of it, then just enough to keep
                  it there, smoothed over ``smooth`` frames. Small motion and detector jitter never shake the crop.
    crop="auto":  "fixed" when the face moves less than ``auto_motion`` x its size over the clip, else "track"."""
    from ..data.face_crop import square_window
    from ..eval.syncnet import S3FDDetector, track_face

    if crop not in ("fixed", "track", "auto"):
        raise ValueError("crop must be 'fixed', 'track' or 'auto'")
    det = detector or _DETECTOR.setdefault("s3fd", S3FDDetector())

    def detect(step):
        b = track_face(det, frames[::step], init_box=bbox)
        if b is None:
            raise RuntimeError("no stable face found; pass bbox=(x0, y0, x1, y1)")
        sz = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        bx, by = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        return b, sz, bx, by, float(np.median(sz)), max(np.ptp(bx), np.ptp(by))

    step = det_every or (1 if crop == "track" else 5)
    boxes, size, cx, cy, face, motion = detect(step)
    t, h, w = frames.shape[:3]
    if crop == "auto":
        crop = "track" if motion > auto_motion * face else "fixed"
        print(f"crop auto: face moves {motion / face:.1f}x its size -> {crop}")
        if crop == "track" and step != (det_every or 1):
            step = det_every or 1
            boxes, size, cx, cy, face, motion = detect(step)
    if crop == "fixed":
        if motion > 0.6 * face:
            print(f"warning: face moves {motion / face:.1f}x its size; try crop='track'")
        x, y, side = square_window(boxes, (w, h), scale, y_shift)
        print(f"face {face:.0f}px -> fixed crop window x={x} y={y} side={side}")
        return FaceWindow([x], [y], side, (w, h))
    # per-frame centre (interpolated between detections), lazy camera with a dead zone, then smoothed
    ts = np.arange(0, t, step)[: len(boxes)]
    dz = deadzone * face
    cx = smooth_track(lazy_follow(smooth_track(np.interp(np.arange(t), ts, cx), 5), dz), smooth)
    cy = smooth_track(lazy_follow(smooth_track(np.interp(np.arange(t), ts, cy), 5), dz), smooth)
    side = int(round(scale * float(np.percentile(size, 90)))) // 2 * 2
    xs = np.round(cx - side / 2).astype(np.int64)
    ys = np.round(cy + y_shift * face - side / 2).astype(np.int64)
    print(f"face {face:.0f}px (moves {motion / face:.1f}x its size) -> tracked crop window side={side}, "
          f"dead zone {deadzone:g}x face, smoothed over {smooth} frames, moves {max(np.ptp(cx), np.ptp(cy)):.0f}px")
    return FaceWindow(xs, ys, side, (w, h))


def crop_faces(frames: np.ndarray, win: FaceWindow, size: int, pad_value: float = 110.0,
               chunk: int = 32) -> np.ndarray:
    """[T, H, W, 3] -> [T, size, size, 3] uint8 crops (grey where the window leaves the frame)."""
    out = np.empty((len(frames), size, size, 3), np.uint8)
    for s in range(0, len(frames), chunk):
        n = min(chunk, len(frames) - s)
        c = torch.full((n, 3, win.side, win.side), pad_value)
        for j in range(n):
            x0, y0, x1, y1, ox, oy = win.box(s + j)
            if x1 > x0 and y1 > y0:
                c[j, :, oy:oy + y1 - y0, ox:ox + x1 - x0] = torch.from_numpy(
                    frames[s + j, y0:y1, x0:x1]).permute(2, 0, 1).float()
        c = F.interpolate(c, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        out[s:s + n] = c.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    return out


def paste_faces(frames: np.ndarray, gen: np.ndarray, win: FaceWindow, masks, feather: float = 0.0,
                color_match: bool = True, chunk: int = 16, src_idx=None) -> np.ndarray:
    """Blend generated crops ``gen`` [N, S, S, 3] into ``frames`` [N, H, W, 3] (in place, also returned).

    ``masks(start, end)`` returns the [n, side, side] blend masks for those output frames; ``src_idx[k]`` is the
    source frame of output frame k (sets the window position when the window moves)."""
    src_idx = np.arange(len(gen)) if src_idx is None else np.asarray(src_idx)
    side = win.side
    for s in range(0, len(gen), chunk):
        n = len(gen[s:s + chunk])
        m_all = masks(s, s + n)
        g_all = torch.from_numpy(gen[s:s + n]).permute(0, 3, 1, 2).float() / 127.5 - 1
        g_all = F.interpolate(g_all, size=(side, side), mode="bicubic", align_corners=False).clamp(-1, 1)
        for j in range(n):
            x0, y0, x1, y1, ox, oy = win.box(int(src_idx[s + j]))
            if x1 <= x0 or y1 <= y0:
                continue
            m = m_all[j:j + 1, oy:oy + y1 - y0, ox:ox + x1 - x0]
            g = g_all[j:j + 1, :, oy:oy + y1 - y0, ox:ox + x1 - x0]
            src = torch.from_numpy(frames[s + j, y0:y1, x0:x1]).permute(2, 0, 1)[None].float() / 127.5 - 1
            comp = composite(g, src, m, feather_px=feather, color_match=color_match)
            frames[s + j, y0:y1, x0:x1] = ((comp[0] + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
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
                 landmarker: FaceLandmarker | None = None, crop: str = "auto", crop_smooth: int = 9,
                 crop_deadzone: float = 0.15, det_every: int | None = None) -> str:
    """Lip-sync ``video`` to ``audio`` through a face crop and write ``out``.

    mode:    "lower" (only the mouth mask is generated) or "full" (the whole crop).
    oval:    "track" (follows the mouth, ``track`` sets its shape) or a dict for a fixed ellipse
             (cx, cy, rx, ry, feather; e.g. ``oval.FIXED_OLD``).
    loop:    reverse | restart | hold | trim, when the audio is longer than the video.
    crop:    "fixed" (one window for the clip), "track" (the window follows the face like a lazy camera; for fast
             head motion) or "auto" (track only when the face moves more than half its size).
             ``crop_deadzone``: how far (x face) the face may drift before the tracked window moves;
             ``crop_smooth``: frames of smoothing; ``det_every``: detect every n-th frame (5 fixed, 1 track).
    preview: write the crop with the mask drawn instead of syncing (no model needed; ``pipe`` may be None).
    cfg:     inference settings; default ``face_infer_config()`` (10 steps, 81-frame windows, no overlap)."""
    if mode not in ("lower", "full"):
        raise ValueError("mode must be 'lower' or 'full'")
    frames, _ = read_video(video, fps=FPS)
    t_src = len(frames)
    win = find_face_window(frames, bbox, scale, detector, det_every, crop, crop_smooth, deadzone=crop_deadzone)
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

    result = paste_faces(frames[idx[:len(res)]].copy(), res, win, paste_masks, feather, src_idx=idx[:len(res)])
    write_video(out, result, FPS, audio_path=audio)
    print(f"{len(result)} frames -> {out}")
    return out
