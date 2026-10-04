"""Pixel-space composite (wilro-sync layer L3).

The decoded window is pasted back into the *original* source frames through a feathered target
mask. Pixels where the mask is 0 are copied from the source unchanged (bit-exact), so non-target
faces and the background cannot be altered by the VAE round trip.
"""

from __future__ import annotations

import torch

from .target import gaussian_blur2d


def feather_mask(mask: torch.Tensor, radius_px: float) -> torch.Tensor:
    """[T, H, W] -> blurred mask, then forced to exactly 0 where the input was 0."""
    if radius_px <= 0:
        return mask.clamp(0, 1)
    blurred = gaussian_blur2d(mask.float(), radius_px / 2.0)
    return (blurred * (mask > 0).float()).clamp(0, 1)


def match_color(gen: torch.Tensor, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Shift the per-frame, per-channel mean of ``gen`` to the source inside a ring around the
    mask boundary (where the two must agree). gen/src: [T, 3, H, W]; mask: [T, H, W] in [0, 1]."""
    band = ((mask > 0.05) & (mask < 0.95)).float().unsqueeze(1)
    denom = band.sum(dim=(2, 3), keepdim=True)
    ok = denom > 16
    delta = ((src - gen) * band).sum(dim=(2, 3), keepdim=True) / denom.clamp_min(1.0)
    return gen + torch.where(ok, delta, torch.zeros_like(delta))


def composite(
    gen: torch.Tensor,
    src: torch.Tensor,
    mask: torch.Tensor,
    feather_px: float = 6.0,
    color_match: bool = True,
) -> torch.Tensor:
    """out = M_f * gen + (1 - M_f) * src. All tensors float, frames [T, 3, H, W], mask [T, H, W]."""
    mf = feather_mask(mask, feather_px)
    if color_match:
        gen = match_color(gen, src, mf)
    m = mf.unsqueeze(1)
    out = m * gen + (1.0 - m) * src
    # bit-exact outside the mask
    return torch.where(m > 0, out, src)
