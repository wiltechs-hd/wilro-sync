"""Replace the background of a (lip-synced) video: segment the person on every frame, composite onto a new
background (an image, a video, a colour, or the original background blurred).

Run it after lip-sync, on the final video: wilro-sync never changes anything outside the mouth, so the order
does not matter for the face, and face detection during lip-sync sees the original video.

The person mask comes from MediaPipe's selfie segmenter (Apache-2.0, fast; hair edges are rough), or from a
mask video you made with another tool (e.g. SAM 2 or BiRefNet for cleaner edges): white = person.
The mask is made firmer (``lo`` / ``hi``), smoothed over time (``smooth`` frames, against flicker) and feathered.
"""

from __future__ import annotations

import os
import re
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from ..io.media import read_video, write_video
from .target import gaussian_blur2d

SEGMENTER_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/"
                 "float16/latest/selfie_segmenter.tflite")
COLORS = {"green": (0, 177, 64), "blue": (0, 71, 187), "white": (255, 255, 255), "black": (0, 0, 0),
          "gray": (128, 128, 128), "grey": (128, 128, 128)}


class SelfieSegmenter:
    """MediaPipe selfie segmenter (model downloaded once into the weights cache). frames -> person probability."""

    def __init__(self, model_path: str | None = None) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("background replacement needs MediaPipe: pip install 'wilro-sync[face]'") from e
        from ..eval.syncnet import weights_dir

        path = model_path or os.path.join(weights_dir(), "selfie_segmenter.tflite")
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            urllib.request.urlretrieve(SEGMENTER_URL, path + ".part")
            os.replace(path + ".part", path)
        opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=path),
                                            running_mode=vision.RunningMode.IMAGE,
                                            output_confidence_masks=True, output_category_mask=False)
        self._mp = mp
        self._seg = vision.ImageSegmenter.create_from_options(opts)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """uint8 [T, H, W, 3] -> float32 [T, H, W] person probability."""
        t, h, w, _ = frames.shape
        out = np.empty((t, h, w), np.float32)
        for i, im in enumerate(frames):
            res = self._seg.segment(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                   data=np.ascontiguousarray(im)))
            masks = [np.asarray(m.numpy_view(), np.float32).reshape(h, w) for m in res.confidence_masks]
            # one mask = person confidence; several (multi-class model) = class 0 is the background
            out[i] = masks[0] if len(masks) == 1 else 1.0 - masks[0]
        return out


def mask_from_video(path: str, num_frames: int, size_hw: tuple[int, int], fps: float) -> np.ndarray:
    """A mask video (white = person) -> float32 [T, H, W] in 0..1, resized and looped/trimmed to the video."""
    m, _ = read_video(path, fps=fps)
    m = torch.from_numpy(m).float().mean(-1)[:, None] / 255.0
    m = F.interpolate(m, size=size_hw, mode="bilinear", align_corners=False)[:, 0].numpy()
    return m[np.arange(num_frames) % len(m)]


def refine_alpha(prob: np.ndarray, lo: float = 0.3, hi: float = 0.7, smooth: int = 3,
                 feather: float = 1.5) -> np.ndarray:
    """Probability -> alpha: ramp between ``lo`` and ``hi``, centred moving average over ``smooth`` frames,
    Gaussian feather of ``feather`` px. [T, H, W] float32."""
    a = np.clip((prob - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    if smooth > 1:
        k = int(smooth)
        pad = np.pad(a, ((k // 2, k - 1 - k // 2), (0, 0), (0, 0)), mode="edge")
        c = np.cumsum(np.concatenate([np.zeros_like(pad[:1]), pad]), axis=0, dtype=np.float64)
        a = ((c[k:] - c[:-k]) / k).astype(np.float32)
    if feather > 0:
        a = gaussian_blur2d(torch.from_numpy(a), feather).clamp(0, 1).numpy()
    return a


def _cover(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize [N, h0, w0, 3] uint8 to fill h x w (scale to cover, centre crop)."""
    n, h0, w0, _ = img.shape
    s = max(h / h0, w / w0)
    nh, nw = max(h, round(h0 * s)), max(w, round(w0 * s))
    x = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    return x[:, :, y0:y0 + h, x0:x0 + w].round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def parse_color(spec: str) -> tuple[int, int, int] | None:
    s = spec.strip().lower()
    if s in COLORS:
        return COLORS[s]
    m = re.fullmatch(r"#?([0-9a-f]{6})", s)
    if m and not os.path.exists(spec):
        v = m.group(1)
        return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    return None


def background_frames(spec: str, frames: np.ndarray, fps: float, blur: float = 25.0) -> np.ndarray:
    """The new background for every frame: ``spec`` = image or video path, a colour (#RRGGBB, green, ...),
    or "blur" (the original background, blurred)."""
    t, h, w, _ = frames.shape
    if spec == "blur":
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        out = np.empty_like(frames)
        for s in range(0, t, 16):
            b = x[s:s + 16]
            blurred = gaussian_blur2d(b.reshape(-1, h, w), blur).reshape(b.shape)
            out[s:s + 16] = blurred.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        return out
    color = parse_color(spec)
    if color is not None:
        return np.broadcast_to(np.array(color, np.uint8), frames.shape)
    if not os.path.isfile(spec):
        raise FileNotFoundError(f"background not found: {spec} (use a file, a colour like #00b140, or 'blur')")
    if spec.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        import imageio.v3 as iio

        img = np.asarray(iio.imread(spec))
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, -1)
        img = _cover(img[None, ..., :3].astype(np.uint8), h, w)[0]
        return np.broadcast_to(img, frames.shape)
    bg, _ = read_video(spec, fps=fps)
    bg = bg[np.arange(t) % len(bg)]  # loop a shorter background video
    out = np.empty_like(frames)
    for s in range(0, t, 32):
        out[s:s + 32] = _cover(bg[s:s + 32], h, w)
    return out


def composite_background(frames: np.ndarray, alpha: np.ndarray, bg: np.ndarray, chunk: int = 32) -> np.ndarray:
    """alpha x frames + (1 - alpha) x bg, all [T, H, W(, 3)]."""
    out = np.empty_like(frames)
    for s in range(0, len(frames), chunk):
        a = alpha[s:s + chunk, ..., None]
        mix = a * frames[s:s + chunk].astype(np.float32) + (1 - a) * bg[s:s + chunk].astype(np.float32)
        out[s:s + chunk] = np.clip(np.round(mix), 0, 255).astype(np.uint8)
    return out


def replace_background(video: str, out: str, bg: str, *, mask_video: str | None = None, segmenter=None,
                       lo: float = 0.3, hi: float = 0.7, smooth: int = 3, feather: float = 1.5,
                       invert: bool = False, blur: float = 25.0, save_mask: str | None = None) -> str:
    """Write ``video`` with its background replaced by ``bg`` (see ``background_frames``), keeping the audio.

    mask_video: use this mask (white = person) instead of segmenting; ``invert`` flips any mask.
    save_mask:  also write the alpha used, as a grey video (to check it or refine it in another tool)."""
    frames, fps = read_video(video)
    t, h, w, _ = frames.shape
    if mask_video:
        prob = mask_from_video(mask_video, t, (h, w), fps)
    else:
        prob = (segmenter or SelfieSegmenter())(frames)
    if invert:
        prob = 1.0 - prob
    alpha = refine_alpha(prob, lo, hi, smooth, feather)
    result = composite_background(frames, alpha, background_frames(bg, frames, fps, blur))
    if save_mask:
        write_video(save_mask, np.repeat((alpha * 255).round().astype(np.uint8)[..., None], 3, -1), fps)
    try:
        write_video(out, result, fps, audio_path=video)
    except RuntimeError:  # the video has no audio track
        write_video(out, result, fps)
    print(f"background {bg!r}: {t} frames -> {out} (person covers {alpha.mean():.0%} of the frame)")
    return out
