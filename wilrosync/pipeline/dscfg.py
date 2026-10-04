"""Dynamic Spatiotemporal Classifier-Free Guidance (OmniSync Sec. 3.4), made target-aware.

    v_cfg = v_uncond + S(x, y, sigma) * (v_cond - v_uncond)
    S     = omega_peak * w(sigma) * G(x, y)

G is the spatial Gaussian of Eq. 7, *normalised* to [omega_base / omega_peak, 1] so that the peak
strength is applied once (the paper's Eq. 9 multiplies two terms that both contain omega_peak).
w(sigma) = sigma ** gamma is Eq. 8 written in the sigma convention (strong early, weak late).

Target-aware additions (wilro-sync):
* the Gaussian is centred on the *target's* mouth;
* outside the target mask the scale is 0 (pure unconditional, i.e. no audio push) when
  ``target_only`` is set;
* inside the target region the scale never drops below ``min_scale`` (default 1.0 = plain
  conditional prediction). Set ``min_scale=0`` for the paper-exact schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DSCFGConfig:
    omega_peak: float = 4.5
    omega_base: float = 1.0
    gamma: float = 1.5
    sigma_rel: float = 0.35  # Gaussian std as a fraction of the face width
    min_scale: float = 1.0
    target_only: bool = True
    normalize_temporal: bool = False  # divide by tau**gamma so the first step uses omega_peak
    enabled: bool = True
    static_scale: float = 4.5  # used when enabled=False (plain CFG ablation)


def spatial_map(
    mouth: torch.Tensor,
    face_width: torch.Tensor,
    latent_hw: tuple[int, int],
    cfg: DSCFGConfig,
) -> torch.Tensor:
    """Normalised spatial guidance map G with values in [omega_base / omega_peak, 1].

    mouth: [F, 2] normalised (x, y); face_width: [F] fraction of frame width. Returns [F, h, w].
    """
    h, w = latent_hw
    dev = mouth.device
    ys = (torch.arange(h, device=dev, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) / w
    mx = mouth[:, 0].view(-1, 1, 1)
    my = mouth[:, 1].view(-1, 1, 1)
    # distances measured in units of frame width so the Gaussian is round in pixel space
    aspect = h / w
    dx = xs.view(1, 1, -1) - mx
    dy = (ys.view(1, -1, 1) - my) * aspect
    sig = (cfg.sigma_rel * face_width).clamp_min(1e-3).view(-1, 1, 1)
    g = torch.exp(-(dx**2 + dy**2) / (2 * sig**2))
    base = cfg.omega_base / cfg.omega_peak
    return base + (1.0 - base) * g


def temporal_weight(sigma: float, cfg: DSCFGConfig, tau: float | None = None) -> float:
    wt = float(sigma) ** cfg.gamma
    if cfg.normalize_temporal and tau:
        wt = wt / (tau**cfg.gamma)
    return wt


def guidance_scale_map(
    sigma: float,
    g_map: torch.Tensor,
    target_mask: torch.Tensor | None,
    cfg: DSCFGConfig,
    tau: float | None = None,
) -> torch.Tensor:
    """Per-position guidance scale S, shape [F, h, w]."""
    if not cfg.enabled:
        s = torch.full_like(g_map, cfg.static_scale)
    else:
        s = cfg.omega_peak * temporal_weight(sigma, cfg, tau) * g_map
        s = s.clamp_min(cfg.min_scale)
    if target_mask is not None and cfg.target_only:
        inside = (target_mask > 0.5).to(s.dtype)
        s = s * inside
    return s


def apply_guidance(v_uncond: torch.Tensor, v_cond: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """v_u + S * (v_c - v_u). ``scale`` is [F, h, w] or [B, F, h, w]; velocities are [B, C, F, h, w]."""
    if scale.ndim == 3:
        scale = scale.unsqueeze(0)
    s = scale.unsqueeze(1).to(v_cond.dtype)
    return v_uncond + s * (v_cond - v_uncond)
