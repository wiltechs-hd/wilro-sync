"""SyncNet audio-visual sync scoring: data filtering (A/V offset + confidence) and LSE-C / LSE-D metrics.

Re-implements the scoring path of `syncnet_python <https://github.com/joonson/syncnet_python>`_ (Chung &
Zisserman, "Out of time: automated lip sync in the wild", 2016) on top of its vendored model definitions and
pretrained weights, without OpenCV / PySceneDetect:

* face track: S3FD detections (every ``det_every`` frames, largest face / IoU-tracked, interpolated, median
  filtered) -> 224x224 crops with the original ``crop_scale=0.4`` rule and grey (110) padding;
* audio: 13-dim MFCCs at 100 Hz (``python_speech_features``) on 16-bit 16 kHz audio;
* 5-frame video windows vs 20-step MFCC windows, distances over shifts of +-``vshift`` frames.

Outputs (per video):
* ``offset``  - A/V offset in frames at 25 fps. The audio that matches video frame ``v`` is at ``v - offset``
                (``offset > 0``: audio leads the lips). Use ``offset_frames = -offset`` when cutting audio windows.
* ``conf``    - median distance minus minimum distance over shifts (= **LSE-C**, higher is better).
* ``min_dist``- mean embedding distance at the best shift (= **LSE-D**, lower is better).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

SYNCNET_FPS = 25
SYNCNET_SR = 16000
WEIGHT_URLS = {
    "syncnet_v2.model": "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
    "sfd_face.pth": "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth",
}


def weights_dir() -> str:
    return os.environ.get("WILROSYNC_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "wilrosync"))


def download_weight(name: str, cache_dir: str | None = None) -> str:
    """Download a pretrained SyncNet / S3FD weight file once and return its local path."""
    cache_dir = cache_dir or weights_dir()
    path = os.path.join(cache_dir, name)
    if not os.path.isfile(path):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(WEIGHT_URLS[name], tmp)
        os.replace(tmp, path)
    return path


@dataclass
class SyncResult:
    offset: int  # frames at 25 fps; audio for video frame v is at v - offset
    conf: float  # LSE-C
    min_dist: float  # LSE-D
    num_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- core math
def sync_offset(im_feat: torch.Tensor, cc_feat: torch.Tensor, vshift: int = 15) -> tuple[int, float, float]:
    """Offset / confidence / min distance from per-window video and audio embeddings ([N, D] each)."""
    n = min(len(im_feat), len(cc_feat))
    if n < 1:
        raise ValueError("not enough frames to score (need at least 6)")
    im, cc = im_feat[:n].float(), cc_feat[:n].float()
    w = 2 * vshift + 1
    ccp = F.pad(cc, (0, 0, vshift, vshift))  # zero-pad in time, like the original
    windows = ccp.unfold(0, w, 1).permute(0, 2, 1)  # [N, w, D]
    dists = F.pairwise_distance(im[:, None, :].expand_as(windows), windows)  # [N, w]
    mdist = dists.mean(0)
    minval, minidx = torch.min(mdist, 0)
    offset = vshift - int(minidx)
    conf = float(torch.median(mdist) - minval)
    return offset, conf, float(minval)


def mfcc(wav: np.ndarray, sr: int = SYNCNET_SR) -> torch.Tensor:
    """[1, 1, 13, T] MFCCs at 100 Hz, computed on 16-bit samples as in syncnet_python."""
    import python_speech_features

    pcm = (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    feats = python_speech_features.mfcc(pcm, sr)  # [T, 13]
    return torch.from_numpy(np.ascontiguousarray(feats.T)).float()[None, None]


def medfilt(x: np.ndarray, k: int = 13) -> np.ndarray:
    """Zero-padded running median (same behaviour as scipy.signal.medfilt)."""
    if len(x) == 0:
        return x
    p = np.pad(np.asarray(x, dtype=np.float64), k // 2)
    return np.median(np.lib.stride_tricks.sliding_window_view(p, k), axis=-1)


def resample_indices(num_frames: int, fps: float, target_fps: float = SYNCNET_FPS) -> np.ndarray:
    if abs(fps - target_fps) < 1e-3:
        return np.arange(num_frames)
    n_out = int(np.floor(num_frames * target_fps / fps))
    return np.clip(np.round(np.arange(n_out) * fps / target_fps).astype(int), 0, num_frames - 1)


def crop_faces(frames: np.ndarray, boxes: np.ndarray, crop_scale: float = 0.4, size: int = 224,
               smooth: int = 13) -> torch.Tensor:
    """syncnet_python's face crop. frames: RGB uint8 [T, H, W, 3]; boxes: [T, 4] (x0, y0, x1, y1).

    Returns BGR float crops [T, 3, size, size] in 0..255 (the network was trained on BGR frames)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    s = np.maximum(boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    if smooth and len(boxes) >= smooth:
        s, cy, cx = medfilt(s, smooth), medfilt(cy, smooth), medfilt(cx, smooth)
    out = torch.empty(len(frames), 3, size, size)
    for i, frame in enumerate(frames):
        bs = max(float(s[i]), 1.0)
        bsi = int(bs * (1 + 2 * crop_scale))
        img = torch.from_numpy(np.ascontiguousarray(frame[..., ::-1])).permute(2, 0, 1).float()  # BGR
        img = F.pad(img, (bsi, bsi, bsi, bsi), value=110.0)
        my, mx = cy[i] + bsi, cx[i] + bsi
        face = img[:, int(my - bs): int(my + bs * (1 + 2 * crop_scale)),
                   int(mx - bs * (1 + crop_scale)): int(mx + bs * (1 + crop_scale))]
        out[i] = F.interpolate(face[None], size=(size, size), mode="bilinear", align_corners=False)[0]
    return out


def whole_frame_crops(frames: np.ndarray, size: int = 224) -> torch.Tensor:
    """For videos that are already tight face crops (e.g. HDTF 512x512): resize the whole frame."""
    x = torch.from_numpy(np.ascontiguousarray(frames[..., ::-1])).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------- face detection / tracking
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class S3FDDetector:
    """S3FD face detector (weights from the syncnet_python release)."""

    MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)[:, None, None]

    def __init__(self, device: str | torch.device | None = None, weights: str | None = None) -> None:
        from ..third_party.s3fd.nets import S3FDNet

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = S3FDNet(device=self.device).to(self.device).eval()
        state = torch.load(weights or download_weight("sfd_face.pth"), map_location=self.device, weights_only=True)
        self.net.load_state_dict(state)

    @torch.no_grad()
    def detect(self, frame: np.ndarray, conf_th: float = 0.9, scale: float = 0.25) -> np.ndarray:
        """frame: RGB uint8 [H, W, 3] -> boxes [N, 5] (x0, y0, x1, y1, score) in pixels."""
        from ..third_party.s3fd.box_utils import nms_

        h, w = frame.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float()[None]  # RGB
        if scale != 1:
            x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        x = x[0].numpy() - self.MEAN  # same preprocessing as syncnet_python (mean applied in RGB order)
        x = torch.from_numpy(np.ascontiguousarray(x[[2, 1, 0]]))[None].to(self.device)  # -> BGR
        det = self.net(x).cpu()
        scale_t = torch.tensor([w, h, w, h], dtype=torch.float32)
        boxes = []
        for i in range(det.size(1)):
            j = 0
            while j < det.size(2) and det[0, i, j, 0] > conf_th:
                boxes.append([*(det[0, i, j, 1:] * scale_t).tolist(), float(det[0, i, j, 0])])
                j += 1
        if not boxes:
            return np.zeros((0, 5))
        boxes = np.asarray(boxes)
        return boxes[nms_(boxes, 0.1)]


def track_face(detector: S3FDDetector, frames: np.ndarray, det_every: int = 1, min_detected: float = 0.5,
               init_box: np.ndarray | None = None, **det_kw) -> np.ndarray | None:
    """One face track over the clip: start from ``init_box`` (target face) or the largest face, then follow
    by IoU. Missing frames are interpolated. Returns [T, 4] or None if the face is found too rarely."""
    t = len(frames)
    keys, kboxes = [], []
    prev = None if init_box is None else np.asarray(init_box, dtype=np.float64)
    for i in range(0, t, max(1, det_every)):
        dets = detector.detect(frames[i], **det_kw)
        if len(dets) == 0:
            continue
        if prev is None:
            k = int(np.argmax((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])))
        else:
            ious = np.array([_iou(prev, d[:4]) for d in dets])
            k = int(np.argmax(ious))
            if ious[k] < 0.3:
                continue
        prev = dets[k, :4]
        keys.append(i)
        kboxes.append(prev)
    n_tried = len(range(0, t, max(1, det_every)))
    if not keys or len(keys) < min_detected * n_tried:
        return None
    kb = np.asarray(kboxes)
    return np.stack([np.interp(np.arange(t), keys, kb[:, c]) for c in range(4)], axis=1)


# ----------------------------------------------------------------------------- scorer
class SyncNet:
    def __init__(self, device: str | torch.device | None = None, weights: str | None = None,
                 detector_weights: str | None = None, load_weights: bool = True) -> None:
        from ..third_party.syncnet_model import S

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = S(num_layers_in_fc_layers=1024).to(self.device).eval()
        if load_weights:
            state = torch.load(weights or download_weight("syncnet_v2.model"), map_location="cpu", weights_only=True)
            own = self.model.state_dict()
            for k, v in state.items():
                own[k].copy_(v)
        self._detector_weights = detector_weights
        self._detector: S3FDDetector | None = None

    @property
    def detector(self) -> S3FDDetector:
        if self._detector is None:
            self._detector = S3FDDetector(self.device, self._detector_weights)
        return self._detector

    @torch.no_grad()
    def embed(self, crops: torch.Tensor, mel: torch.Tensor, batch_size: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """crops: BGR float [T, 3, 224, 224] (0..255) at 25 fps; mel: [1, 1, 13, T_m] at 100 Hz."""
        n = min(len(crops), mel.shape[-1] // 4)
        last = n - 5
        if last < 1:
            raise ValueError("clip too short for SyncNet (needs > 5 frames)")
        vid = crops.permute(1, 0, 2, 3)[None]  # [1, 3, T, 224, 224]
        im_feat, cc_feat = [], []
        for i in range(0, last, batch_size):
            idx = range(i, min(last, i + batch_size))
            im = torch.cat([vid[:, :, v: v + 5] for v in idx]).to(self.device)
            cc = torch.cat([mel[:, :, :, v * 4: v * 4 + 20] for v in idx]).to(self.device)
            im_feat.append(self.model.forward_lip(im).cpu())
            cc_feat.append(self.model.forward_aud(cc).cpu())
        return torch.cat(im_feat), torch.cat(cc_feat)

    def score_crops(self, crops: torch.Tensor, wav: np.ndarray, vshift: int = 15) -> SyncResult:
        n_audio = int(len(wav) / SYNCNET_SR * SYNCNET_FPS)
        crops = crops[: min(len(crops), n_audio)]
        im, cc = self.embed(crops, mfcc(wav))
        offset, conf, dist = sync_offset(im, cc, vshift)
        return SyncResult(offset, conf, dist, len(crops))

    def score_video(self, frames: np.ndarray, wav: np.ndarray, fps: float = SYNCNET_FPS, *,
                    boxes: np.ndarray | None = None, bbox=None, assume_cropped: bool = False,
                    det_every: int = 1, vshift: int = 15) -> SyncResult | None:
        """Score one face. frames: RGB uint8 [T, H, W, 3]; wav: mono float32 at 16 kHz.

        * ``boxes`` [T, 4]: use this face track directly;
        * ``bbox`` (x0, y0, x1, y1): track the face starting from this box (target face in multi-face scenes);
        * ``assume_cropped``: the video is already a face crop, skip detection;
        * otherwise the largest face is detected and tracked.
        Returns None when no face track is found."""
        idx = resample_indices(len(frames), fps)
        frames = frames[idx]
        if assume_cropped:
            crops = whole_frame_crops(frames)
        else:
            if boxes is None:
                boxes = track_face(self.detector, frames, det_every=det_every, init_box=bbox)
                if boxes is None:
                    return None
            else:
                boxes = np.asarray(boxes)[idx] if np.ndim(boxes) == 2 else np.tile(boxes, (len(frames), 1))
            crops = crop_faces(frames, boxes)
        return self.score_crops(crops, wav, vshift)

    def score_file(self, video: str, audio: str | None = None, max_seconds: float | None = None, **kw) -> SyncResult | None:
        """Read a video (resampled to 25 fps) and its own or a separate audio track, then ``score_video``."""
        from ..io.media import read_audio, read_video

        max_frames = int(max_seconds * SYNCNET_FPS) if max_seconds else None
        frames, fps = read_video(video, max_frames=max_frames, fps=SYNCNET_FPS)
        wav = read_audio(audio or video)
        wav = wav[: int(len(frames) / fps * SYNCNET_SR)]
        return self.score_video(frames, wav, fps, **kw)


def lse(video: str, audio: str | None = None, bbox=None, scorer: SyncNet | None = None, **kw) -> dict:
    """LSE-C / LSE-D of one (generated) video; pass ``bbox`` to score only the target face."""
    scorer = scorer or SyncNet()
    r = scorer.score_file(video, audio, bbox=bbox, **kw)
    if r is None:
        return {"lse_c": float("nan"), "lse_d": float("nan"), "offset": None, "num_frames": 0}
    return {"lse_c": r.conf, "lse_d": r.min_dist, "offset": r.offset, "num_frames": r.num_frames}
