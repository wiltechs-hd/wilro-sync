"""Face-centred square crops for training clips (milestone M3).

Like HDTF's "method 2": detect the face on a low-resolution, low-fps copy of the clip (S3FD, tracked by IoU),
merge the per-frame boxes into **one fixed square window** per clip (median centre, enlarged size), then let
ffmpeg crop + resize + resample to 25 fps in a single pass (audio is kept). A fixed window avoids crop jitter,
which the model would otherwise learn as head motion.
"""

from __future__ import annotations

import numpy as np

from ..io.media import ffmpeg, probe_video, read_video


def square_window(boxes: np.ndarray, frame_wh: tuple[int, int], scale: float = 1.8,
                  y_shift: float = 0.1) -> tuple[int, int, int] | None:
    """One square (x, y, side) covering the tracked face boxes [T, 4] in source pixels.

    side = ``scale`` x the 90th-percentile face size (plus the spread of the centre), centred on the median face
    centre, moved down by ``y_shift`` x face size so mouth and chin stay well inside the crop."""
    boxes = np.asarray(boxes, dtype=np.float64)
    size = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    face = float(np.percentile(size, 90))
    spread = float(max(np.ptp(cx), np.ptp(cy)))
    side = int(round(scale * face + spread)) // 2 * 2
    if side < 16:
        return None
    mx, my = float(np.median(cx)), float(np.median(cy)) + y_shift * face
    return int(round(mx - side / 2)), int(round(my - side / 2)), side


def crop_filter(window: tuple[int, int, int], frame_wh: tuple[int, int], size: int) -> str:
    """ffmpeg filter for the window; pads with grey when the window leaves the frame."""
    x, y, side = window
    w, h = frame_wh
    pad = max(0, -x, -y, x + side - w, y + side - h)
    chain = []
    if pad:
        pad = int(pad) + 2
        chain.append(f"pad={w + 2 * pad}:{h + 2 * pad}:{pad}:{pad}:color=0x6e6e6e")
        x, y = x + pad, y + pad
    chain.append(f"crop={side}:{side}:{x}:{y}")
    chain.append(f"scale={size}:{size}:flags=lanczos")
    return ",".join(chain)


def face_crop_video(src: str, dst: str, detector=None, size: int = 512, scale: float = 1.8, fps: float = 25.0,
                    det_fps: float = 5.0, det_max_side: int = 640, min_face_px: int = 96,
                    max_center_motion: float = 0.6, min_detected: float = 0.6) -> dict | None:
    """Write a ``size`` x ``size`` face-centred, ``fps`` copy of ``src`` (with audio) to ``dst``.

    Returns crop info, or None when the clip is unusable: no stable face, face smaller than ``min_face_px``
    source pixels, or the face moves more than ``max_center_motion`` x its size (the fixed crop would cut it).
    """
    from ..eval.syncnet import S3FDDetector, track_face

    detector = detector or S3FDDetector()
    info = probe_video(src)
    small, _ = read_video(src, fps=det_fps, max_side=det_max_side)
    r = info["width"] / small.shape[2]  # small -> source pixels
    boxes = track_face(detector, small, det_every=1, min_detected=min_detected, scale=1.0)
    if boxes is None:
        return None
    boxes = boxes * r
    face = float(np.median(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])))
    if face < min_face_px:
        return None
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    if max(np.ptp(cx), np.ptp(cy)) > max_center_motion * face:
        return None
    window = square_window(boxes, (info["width"], info["height"]), scale)
    if window is None:
        return None
    vf = crop_filter(window, (info["width"], info["height"]), size)
    ffmpeg("-i", src, "-vf", vf, "-r", str(fps), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ar", "16000", "-ac", "1", dst)
    return {"window": list(window), "face_px": round(face, 1), "src_wh": [info["width"], info["height"]]}
