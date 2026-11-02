"""Mouth masks for face-crop lip sync (``mode="lower"``).

Two kinds of mask, both in face-crop coordinates (0..1, the mouth sits at about (0.5, 0.6)):

* **tracked oval** (default): follows the mouth every frame using MediaPipe face landmarks. Position and head
  tilt are per frame; the size is fixed for the whole clip (widest mouth, widest jaw opening), so the mask
  never pulses. The shape is an egg: a flat top between the nose and the upper lip and a round bottom past
  the chin. Optional ``spread`` adds a wide outer soft edge, ``spread_up`` lets it reach over the nose.
* **fixed oval**: a static soft ellipse (``lower_face_mask``).

Parameters were tuned with SyncNet LSE-C on held-out and training clips; see docs/inference_face.md.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .target import gaussian_blur2d

# MediaPipe face-mesh landmark ids
_EYE_L, _EYE_R = 33, 263  # outer eye corners (head tilt)
_MOUTH_L, _MOUTH_R = 61, 291  # mouth corners
_SUBNASALE, _CHIN = 2, 152  # bottom of the nose, bottom of the chin
_LIP_TOP = 0  # top of the upper lip
LANDMARK_IDS = (_EYE_L, _EYE_R, _MOUTH_L, _MOUTH_R, _SUBNASALE, _CHIN, _LIP_TOP)
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")

# the static ellipse used before the tracked oval (kept for comparisons)
FIXED_OLD = dict(cx=0.5, cy=0.64, rx=0.32, ry=0.20, feather=0.04)
FIXED_DEFAULT = dict(cx=0.5, cy=0.68, rx=0.30, ry=0.18, feather=0.025)


@dataclass
class TrackConfig:
    """Tracked-oval shape. Lengths are fractions of the nose-to-chin distance D unless noted."""

    width: float = 2.2  # half-width = width x half the widest mouth width in the clip
    bottom: float = 0.4  # room below the chin, on top of the widest jaw opening (the parameter that matters)
    top: float = 0.3  # top edge between the bottom of the nose (0) and the top of the upper lip (1)
    feather: float = 0.04  # edge softness; capped so the soft edge stays between the nose and the lip
    spread: float = 0.0  # extra-wide outer soft edge (sides and below); 0 = off
    spread_up: float = 0.0  # how high the spread may reach: 0 = bottom of the nose, 1 = just under the eyes
    smooth: int = 5  # centred moving average of the landmarks, in frames

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackedOval:
    """Per-frame oval: ``params`` [T, 7] = (cx, cy, rx, ry_top, ry_bottom, angle, spread limit) in crop units,
    ``mouth`` [T, 2] mouth centres, ``sigma`` (tight edge, spread) in crop units."""

    params: np.ndarray
    mouth: np.ndarray
    sigma: tuple[float, float]

    def select(self, idx) -> TrackedOval:
        return TrackedOval(self.params[idx], self.mouth[idx], self.sigma)

    def masks(self, size: int, idx=None) -> torch.Tensor:
        p = self.params if idx is None else self.params[idx]
        return oval_masks(p, size, self.sigma)


# ----------------------------------------------------------------------------- landmarks
class FaceLandmarker:
    """MediaPipe Face Landmarker (model downloaded once into the weights cache)."""

    def __init__(self, model_path: str | None = None, num_faces: int = 2) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("the tracked oval needs MediaPipe: pip install 'wilro-sync[face]'") from e
        from ..eval.syncnet import weights_dir

        path = model_path or os.path.join(weights_dir(), "face_landmarker.task")
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            urllib.request.urlretrieve(LANDMARKER_URL, path + ".part")
            os.replace(path + ".part", path)
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=path),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=num_faces)
        self._mp = mp
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, crops: np.ndarray) -> np.ndarray:
        """[T, S, S, 3] uint8 face crops -> [T, 7, 2] landmarks in 0..1 (``LANDMARK_IDS`` order), NaN = missed.
        With two faces in a crop, the one whose nose is nearest the crop centre (the target) is used."""
        out = np.full((len(crops), len(LANDMARK_IDS), 2), np.nan, np.float32)
        for t, im in enumerate(crops):
            res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                 data=np.ascontiguousarray(im)))
            if not res.face_landmarks:
                continue
            faces = [np.array([[f[i].x, f[i].y] for i in LANDMARK_IDS], np.float32) for f in res.face_landmarks]
            out[t] = min(faces, key=lambda p: float(np.abs(p[4] - 0.5).sum()))
        return out


def fill_and_smooth(points: np.ndarray, k: int) -> np.ndarray:
    """Interpolate missed frames (NaN), then a centred moving average of ``k`` frames."""
    t = np.arange(len(points))
    flat = points.reshape(len(points), -1).astype(np.float64)
    ok = ~np.isnan(flat[:, 0])
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(t, t[ok], flat[ok, c])
    if k > 1:
        pad = np.pad(flat, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
        flat = np.stack([np.convolve(pad[:, c], np.ones(k) / k, "valid") for c in range(flat.shape[1])], 1)
    return flat.reshape(points.shape).astype(np.float32)


# ----------------------------------------------------------------------------- tracked oval
def oval_from_landmarks(landmarks: np.ndarray, cfg: TrackConfig | None = None,
                        min_found: float = 0.5) -> TrackedOval | None:
    """Build the per-frame oval from [T, 7, 2] landmarks (see ``FaceLandmarker``); None if the face is found
    in fewer than ``min_found`` of the frames."""
    o = cfg or TrackConfig()
    hit = ~np.isnan(landmarks[:, 0, 0])
    if hit.mean() < min_found:
        return None
    p = fill_and_smooth(landmarks, o.smooth)
    eye_l, eye_r, m_l, m_r, sub, chin, lip = (p[:, i] for i in range(7))
    d = eye_r - eye_l
    ang = np.arctan2(d[:, 1], d[:, 0])  # head roll
    cos, sin = np.cos(ang), np.sin(ang)

    def rot(v):  # into the head's upright frame
        return np.stack([v[:, 0] * cos + v[:, 1] * sin, -v[:, 0] * sin + v[:, 1] * cos], 1)

    mouth = (m_l + m_r) / 2
    dist = np.linalg.norm(chin - sub, axis=1)  # nose-to-chin, grows when the jaw opens
    d_max = float(np.percentile(dist[hit] if hit.sum() > 3 else dist, 95))
    half_w = float(np.percentile(np.linalg.norm(m_r - m_l, axis=1), 95)) / 2
    gap = float(np.percentile(rot(lip)[:, 1] - rot(sub)[:, 1], 5))  # nose to upper lip (smallest)
    top = rot(sub)[:, 1] + o.top * gap
    sigma = min(o.feather * d_max, min(o.top, 1 - o.top) * gap / 2.5)  # soft edge stays between nose and lip
    bottom = rot(sub)[:, 1] + d_max * (1 + o.bottom)
    cx_u = rot(mouth)[:, 0]
    cy_u = np.maximum(rot(mouth)[:, 1], rot(lip)[:, 1])  # widest point: mouth corners / upper lip line
    cx, cy = cx_u * cos - cy_u * sin, cx_u * sin + cy_u * cos  # back to crop coordinates
    rx = np.full(len(p), max(o.width * half_w, 0.6 * d_max))
    nose = rot(sub)[:, 1] - cy_u
    eyes = rot((eye_l + eye_r) / 2)[:, 1] + 0.15 * d_max - cy_u  # a little under the eyes
    limit = nose + float(np.clip(o.spread_up, 0, 1)) * (eyes - nose)
    params = np.stack([cx, cy, rx, cy_u - top, bottom - cy_u, ang, limit], 1).astype(np.float32)
    return TrackedOval(params, mouth.astype(np.float32), (float(sigma), float(o.spread * d_max)))


def track_oval(crops: np.ndarray, cfg: TrackConfig | None = None,
               landmarker: FaceLandmarker | None = None) -> TrackedOval | None:
    """Landmarks on every crop -> ``TrackedOval`` (None if the face is rarely found)."""
    return oval_from_landmarks((landmarker or FaceLandmarker())(crops), cfg)


def oval_masks(params, size: int, sigma) -> torch.Tensor:
    """Render [T, size, size] soft egg shapes from ``TrackedOval.params``.

    A flat half-superellipse above the mouth line (up to below the nose) and a deep half-ellipse below it (past
    the chin). ``sigma = (tight, spread)``: the tight soft edge everywhere, plus an optional wide one (spread)
    reaching into the cheeks, jaw and neck, cut off at the spread limit line (nose .. eyes)."""
    sig_t, sig_w = sigma if isinstance(sigma, (tuple, list)) else (sigma, 0.0)
    pr = torch.as_tensor(np.asarray(params), dtype=torch.float32)
    ys = ((torch.arange(size, dtype=torch.float32) + 0.5) / size).view(1, -1, 1)
    xs = ((torch.arange(size, dtype=torch.float32) + 0.5) / size).view(1, 1, -1)
    cx, cy, rx, ry_t, ry_b, a = (pr[:, i].view(-1, 1, 1) for i in range(6))
    dx, dy = xs - cx, ys - cy
    u = dx * torch.cos(a) + dy * torch.sin(a)
    v = -dx * torch.sin(a) + dy * torch.cos(a)
    upper = (u / rx).abs() ** 4 + (v / ry_t).abs() ** 4  # flat top: covers smile corners
    lower = (u / rx) ** 2 + (v / ry_b) ** 2  # round bottom: jaw and chin
    m = (torch.where(v < 0, upper, lower) <= 1.0).float()
    tight = gaussian_blur2d(m, sig_t * size).clamp(0, 1)
    if sig_w <= sig_t:
        return tight
    wide = (gaussian_blur2d(m, sig_w * size) * 2.0).clamp(0, 1)  # x2: full strength at the shape's edge
    limit = pr[:, 6].view(-1, 1, 1) if pr.shape[1] > 6 else torch.full_like(cy, -1.0)
    below = torch.sigmoid((v - limit) / max(sig_w / 3, sig_t, 1e-3) * 1.7)  # 0 above the limit line
    return torch.maximum(tight, wide * below).clamp(0, 1)


# ----------------------------------------------------------------------------- fixed oval
def lower_face_mask(size: int, cx: float = 0.5, cy: float = 0.68, rx: float = 0.30, ry: float = 0.18,
                    feather: float = 0.025) -> torch.Tensor:
    """Static soft ellipse [size, size] in crop coordinates."""
    ys = (torch.arange(size, dtype=torch.float32).view(-1, 1) + 0.5) / size
    xs = (torch.arange(size, dtype=torch.float32).view(1, -1) + 0.5) / size
    m = ((((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0).float()
    return gaussian_blur2d(m[None], feather * size)[0].clamp(0, 1)
