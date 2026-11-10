"""Perceptual (LPIPS) loss on the decoded mouth area, for sharper mouths.

Training runs on pre-encoded latents, so this decodes a small piece of both the model's guess of the clean
latents and the true latents with the frozen Wan VAE, and compares them with LPIPS:

    z0_hat = x_sigma - sigma * v                      (the model's one-step guess of the clean latents)
    loss  += weight * LPIPS(decode(crop(z0_hat)), decode(crop(z_ab)))

Only for samples with sigma < ``lpips_max_sigma``: at high noise the one-step guess is an average of many
possible mouths (blurry by design), and LPIPS would push the model to guess sharp details it cannot know.
To fit next to the model on a 24 GB GPU, only a mouth crop (``lpips_crop`` latent cells) of a few latent frames
(``lpips_latent_frames``) is decoded; gradients flow through the VAE decoder into ``z0_hat``.
Both sides go through the same decoder, so only the model's error is measured, not the VAE's.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_lpips(net: str = "alex") -> nn.Module:
    """LPIPS network (frozen). Images in [-1, 1], [N, 3, H, W] -> [N, 1, 1, 1]."""
    try:
        import lpips
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("the LPIPS loss needs the lpips package: pip install 'wilro-sync[train]'") from e
    m = lpips.LPIPS(net=net, verbose=False).eval()
    m.requires_grad_(False)
    return m


def mouth_crop_boxes(mouth: torch.Tensor, latent_hw: tuple[int, int], size: int) -> list[tuple[int, int, int]]:
    """Square crop (y0, x0, side) in latent cells around the mean mouth position of each sample.

    mouth: [B, F, 2] normalised (x, y). The crop is clamped to the latent frame."""
    h, w = latent_hw
    side = int(min(size, h, w))
    out = []
    for m in mouth:
        cx, cy = m[:, 0].float().mean().item(), m[:, 1].float().mean().item()
        x0 = int(round(cx * w - side / 2))
        y0 = int(round(cy * h - side / 2))
        out.append((min(max(0, y0), h - side), min(max(0, x0), w - side), side))
    return out


class LatentLPIPS(nn.Module):
    def __init__(self, vae: nn.Module, net: nn.Module, crop: int = 24, latent_frames: int = 2,
                 max_sigma: float = 0.3) -> None:
        super().__init__()
        self.vae, self.net = vae, net
        self.crop, self.latent_frames, self.max_sigma = crop, latent_frames, max_sigma
        self.last: float | None = None  # last value, for logging

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Normalised latents [B, C, F, h, w] -> frames [B*T, 3, H, W] in [-1, 1], differentiable."""
        v = self.vae
        x = z.float() * v.std + v.mean
        frames = v.vae.decode(x.to(v.dtype), return_dict=False)[0].float().clamp(-1, 1)  # [B, 3, T, H, W]
        b, c, t, hh, ww = frames.shape
        return frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, hh, ww)

    def forward(self, z0_hat: torch.Tensor, z_true: torch.Tensor, sigma: torch.Tensor, mouth: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor | None:
        """Mean LPIPS over the low-noise samples of the batch, or None if there are none."""
        keep = (sigma.view(-1) < self.max_sigma).nonzero().flatten().tolist()
        if not keep:
            self.last = None
            return None
        f = z0_hat.shape[2]
        n = int(min(self.latent_frames, f))
        boxes = mouth_crop_boxes(mouth[keep], tuple(z0_hat.shape[-2:]), self.crop)
        losses = []
        for i, (y0, x0, s) in zip(keep, boxes):
            t0 = int(torch.randint(0, f - n + 1, (1,), generator=generator).item())
            pred = z0_hat[i:i + 1, :, t0:t0 + n, y0:y0 + s, x0:x0 + s]
            with torch.no_grad():
                real = self._decode(z_true[i:i + 1, :, t0:t0 + n, y0:y0 + s, x0:x0 + s])
            losses.append(self.net(self._decode(pred), real).mean())
        loss = torch.stack(losses).mean()
        self.last = float(loss.detach())
        return loss
