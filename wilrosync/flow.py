"""Flow-matching utilities.

Convention used everywhere in wilro-sync (same as Wan / diffusers' FlowMatch schedulers):

    x_sigma = (1 - sigma) * x_data + sigma * eps,      sigma = 1 -> pure noise, sigma = 0 -> data
    velocity v = d x_sigma / d sigma = eps - x_data
    model timestep t = sigma * 1000

The OmniSync paper mixes two conventions (Eq. 2 uses t = 1 for data, Eq. 5 uses tau as a noise
level). In this code tau == sigma_start, and the paper's ``t > 850`` threshold is ``sigma > 0.85``.
"""

from __future__ import annotations

import torch

NUM_TRAIN_TIMESTEPS = 1000


def _bcast(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    sigma = torch.as_tensor(sigma, device=like.device, dtype=torch.float32)
    while sigma.ndim < like.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def add_noise(x_data: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Linear interpolation path ``(1 - sigma) * x + sigma * eps`` (paper Eq. 5, FM_add)."""
    s = _bcast(sigma, x_data)
    return ((1.0 - s) * x_data.float() + s * noise.float()).to(x_data.dtype)


def target_velocity(x_data: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - x_data


def euler_step(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    return (x.float() + (sigma_next - sigma) * v.float()).to(x.dtype)


def sigma_to_timestep(sigma: torch.Tensor | float) -> torch.Tensor:
    return torch.as_tensor(sigma, dtype=torch.float32) * NUM_TRAIN_TIMESTEPS


def shift_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    """Wan/SD3 timestep shift: maps uniform u in [0, 1] towards higher noise when shift > 1."""
    if shift == 1.0:
        return u
    return shift * u / (1.0 + (shift - 1.0) * u)


def unshift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    if shift == 1.0:
        return sigma
    return sigma / (shift - (shift - 1.0) * sigma)


def inference_sigmas(num_steps: int, tau: float = 0.92, shift: float = 3.0) -> torch.Tensor:
    """Progressive-noise-init schedule: ``num_steps`` Euler steps from sigma = tau down to 0.

    The schedule is the usual shifted schedule, truncated so that it starts exactly at ``tau``
    (paper Sec. 3.3, Eq. 6). Returns ``num_steps + 1`` values, first == tau, last == 0.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    u_start = unshift_sigma(torch.tensor(tau, dtype=torch.float64), shift)
    u = torch.linspace(float(u_start), 0.0, num_steps + 1, dtype=torch.float64)
    sig = shift_sigma(u, shift)
    sig[0] = tau
    sig[-1] = 0.0
    return sig.float()


def sample_training_sigmas(
    batch_size: int,
    *,
    mode: str = "shifted_uniform",
    shift: float = 3.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    generator: torch.Generator | None = None,
    min_sigma: float = 1e-3,
) -> torch.Tensor:
    """Draw per-sample training sigmas.

    ``shifted_uniform``: u ~ U(0, 1), sigma = shift(u). ``logit_normal``: u = sigmoid(N(m, s)), then shift.
    """
    if mode == "shifted_uniform":
        u = torch.rand(batch_size, generator=generator)
    elif mode == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, generator=generator) * logit_std + logit_mean)
    else:
        raise ValueError(f"unknown sigma sampling mode {mode!r}")
    return shift_sigma(u, shift).clamp(min_sigma, 1.0)
