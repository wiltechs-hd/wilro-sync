"""Target Region Lock (wilro-sync layer L2).

After every denoising step the latent outside the target mask is replaced by the *source* latent
re-noised to the current noise level, using the same noise sample as the progressive noise init.
Non-target faces and the background therefore stay on the source trajectory, and only the target
region is free to change. During the last ``release_steps`` steps the mask is dilated by
``release_dilate`` latent cells so the boundary can blend with its surroundings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..flow import add_noise


def dilate_mask(mask: torch.Tensor, cells: int) -> torch.Tensor:
    """Spatial max-pool dilation of a [..., F, h, w] mask by ``cells`` latent cells."""
    if cells <= 0:
        return mask
    shp = mask.shape
    m = mask.reshape(-1, 1, shp[-2], shp[-1]).float()
    m = F.max_pool2d(m, kernel_size=2 * cells + 1, stride=1, padding=cells)
    return m.reshape(shp).to(mask.dtype)


class RegionLock:
    def __init__(
        self,
        z_src: torch.Tensor,
        noise: torch.Tensor,
        mask_lat: torch.Tensor,
        release_steps: int = 3,
        release_dilate: int = 1,
    ) -> None:
        """z_src, noise: [B, C, F, h, w]; mask_lat: [F, h, w] or [B, F, h, w], 1 = editable."""
        if mask_lat.ndim == 3:
            mask_lat = mask_lat.unsqueeze(0)
        self.z_src = z_src
        self.noise = noise
        self.mask = mask_lat.unsqueeze(1).to(z_src.dtype)  # [B, 1, F, h, w]
        self.mask_release = dilate_mask(mask_lat, release_dilate).unsqueeze(1).to(z_src.dtype)
        self.release_steps = release_steps

    def __call__(self, x: torch.Tensor, sigma_next: float, step_idx: int, num_steps: int) -> torch.Tensor:
        released = step_idx >= num_steps - self.release_steps
        m = self.mask_release if released else self.mask
        ref = add_noise(self.z_src, self.noise, sigma_next)
        return m * x + (1.0 - m) * ref
