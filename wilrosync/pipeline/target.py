"""Per-frame target-face signals and their latent-grid versions.

The full Target-Face Module (detection, identity matching, SAM2 tracking, active-speaker detection)
is milestone M1. This file defines the interface every target provider produces, plus two
providers that need no detector: the whole frame (single-speaker videos) and a fixed bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def latent_frame_groups(num_frames: int, temporal_factor: int = 4) -> list[list[int]]:
    """Video-frame indices covered by each latent frame of Wan's causal 3D VAE.

    Latent frame 0 <- video frame 0; latent frame k >= 1 <- video frames 4k-3 .. 4k.
    """
    if (num_frames - 1) % temporal_factor != 0:
        raise ValueError(f"num_frames must be 1 + {temporal_factor}k, got {num_frames}")
    groups = [[0]]
    for k in range(1, (num_frames - 1) // temporal_factor + 1):
        groups.append(list(range(temporal_factor * k - temporal_factor + 1, temporal_factor * k + 1)))
    return groups


def gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the last two dims of a [N, H, W] tensor."""
    if sigma <= 0:
        return x
    radius = max(1, int(3 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-(t**2) / (2 * sigma**2))
    k = (k / k.sum()).to(x.dtype)
    y = x.unsqueeze(1)
    y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    return y.squeeze(1)


@dataclass
class TargetSignals:
    """Per-video-frame target information in pixel space.

    mask:       [T, H, W] float in [0, 1]; 1 = region the model may edit (target face + jaw)
    mouth:      [T, 2] mouth center (x, y), normalised to [0, 1]
    face_width: [T] face width as a fraction of frame width
    presence:   [T] bool, whether the target is visible
    """

    mask: torch.Tensor
    mouth: torch.Tensor
    face_width: torch.Tensor
    presence: torch.Tensor
    full_frame: bool = False

    @property
    def num_frames(self) -> int:
        return self.mask.shape[0]

    # ----------------------------------------------------------------- providers
    @classmethod
    def whole_frame(
        cls, num_frames: int, height: int, width: int, mouth=(0.5, 0.62), face_width: float = 0.45
    ) -> TargetSignals:
        """Single-speaker mode: everything is editable (exact OmniSync behaviour)."""
        return cls(
            mask=torch.ones(num_frames, 8, 8),  # resolution-free; resized on demand
            mouth=torch.tensor(mouth, dtype=torch.float32).expand(num_frames, 2).clone(),
            face_width=torch.full((num_frames,), face_width),
            presence=torch.ones(num_frames, dtype=torch.bool),
            full_frame=True,
        )

    @classmethod
    def from_bbox(
        cls,
        num_frames: int,
        height: int,
        width: int,
        bbox: tuple[float, float, float, float],
        dilate: float = 0.15,
        jaw_extend: float = 0.15,
        feather_px: float | None = None,
        mask_max_side: int = 512,
    ) -> TargetSignals:
        """Fixed face box (x0, y0, x1, y1) in pixels. Works for any character, no detector needed.

        The mask is stored at a reduced resolution (max side ``mask_max_side``) to keep long videos
        cheap; it is resized on demand."""
        x0, y0, x1, y1 = bbox
        mouth = torch.tensor([(x0 + x1) / 2 / width, (y0 + 0.78 * (y1 - y0)) / height], dtype=torch.float32)
        face_w = (x1 - x0) / width
        r = min(1.0, mask_max_side / max(height, width))
        height, width = max(1, round(height * r)), max(1, round(width * r))
        x0, y0, x1, y1 = (v * r for v in bbox)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            raise ValueError(f"invalid bbox {bbox}")
        ex0, ex1 = x0 - dilate * bw, x1 + dilate * bw
        ey0, ey1 = y0 - dilate * bh, y1 + (dilate + jaw_extend) * bh
        ys = torch.arange(height, dtype=torch.float32).view(-1, 1) + 0.5
        xs = torch.arange(width, dtype=torch.float32).view(1, -1) + 0.5
        # soft ellipse covering the dilated box
        cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
        rx, ry = (ex1 - ex0) / 2, (ey1 - ey0) / 2
        d = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
        mask = (d <= 1.0).float()
        fp = feather_px if feather_px is not None else 0.06 * max(bw, bh)
        mask = gaussian_blur2d(mask[None], fp)[0].clamp(0, 1)
        return cls(
            mask=mask.expand(num_frames, height, width),
            mouth=mouth.expand(num_frames, 2).clone(),
            face_width=torch.full((num_frames,), face_w),
            presence=torch.ones(num_frames, dtype=torch.bool),
        )

    # ----------------------------------------------------------------- slicing / resizing
    def slice(self, start: int, end: int) -> TargetSignals:
        return TargetSignals(
            self.mask[start:end], self.mouth[start:end], self.face_width[start:end],
            self.presence[start:end], self.full_frame,
        )

    def select(self, idx) -> TargetSignals:
        idx = torch.as_tensor(idx, dtype=torch.long)
        return TargetSignals(
            self.mask[idx], self.mouth[idx], self.face_width[idx], self.presence[idx], self.full_frame
        )

    def pad_to(self, length: int) -> TargetSignals:
        """Repeat the last frame until ``length`` frames (used for short windows)."""
        n = self.num_frames
        if n >= length:
            return self
        idx = torch.cat([torch.arange(n), torch.full((length - n,), n - 1)])
        return TargetSignals(
            self.mask[idx], self.mouth[idx], self.face_width[idx], self.presence[idx], self.full_frame
        )

    def resized_mask(self, height: int, width: int) -> torch.Tensor:
        if self.mask.shape[-2:] == (height, width):
            return self.mask
        return F.interpolate(self.mask[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]

    # ----------------------------------------------------------------- latent grid
    def to_latent(
        self, latent_hw: tuple[int, int], temporal_factor: int = 4
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Latent-grid versions: mask [F, h, w], mouth [F, 2], face_width [F].

        A latent cell is editable if the target touches it in *any* of the frames it covers (max over
        the temporal group), and the spatial value is the area fraction (adaptive average pool).
        """
        h, w = latent_hw
        groups = latent_frame_groups(self.num_frames, temporal_factor)
        m = F.adaptive_avg_pool2d(self.mask[:, None].float(), (h, w))[:, 0]
        mask_lat = torch.stack([m[g].amax(0) for g in groups])
        mouth_lat = torch.stack([self.mouth[g].mean(0) for g in groups])
        fw_lat = torch.stack([self.face_width[g].mean() for g in groups])
        return mask_lat.clamp(0, 1), mouth_lat, fw_lat
