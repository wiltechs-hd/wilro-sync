import math

import torch

from wilrosync.flow import (
    add_noise,
    euler_step,
    inference_sigmas,
    sample_training_sigmas,
    shift_sigma,
    target_velocity,
    unshift_sigma,
)
from wilrosync.pipeline.dscfg import DSCFGConfig, apply_guidance, guidance_scale_map, spatial_map
from wilrosync.pipeline.region_lock import RegionLock, dilate_mask


def test_flow_path_and_exact_euler():
    g = torch.Generator().manual_seed(0)
    x0, eps = torch.randn(2, 4, 3, 5, 5, generator=g), torch.randn(2, 4, 3, 5, 5, generator=g)
    v = target_velocity(x0, eps)
    sig = inference_sigmas(10, tau=0.92, shift=3.0)
    x = add_noise(x0, eps, sig[0])
    for s, s1 in zip(sig[:-1].tolist(), sig[1:].tolist()):
        x = euler_step(x, v, s, s1)  # exact velocity -> lands on the data
    torch.testing.assert_close(x, x0, atol=1e-5, rtol=1e-5)


def test_inference_schedule():
    sig = inference_sigmas(50, tau=0.92, shift=3.0)
    assert len(sig) == 51 and math.isclose(sig[0].item(), 0.92, abs_tol=1e-6) and sig[-1] == 0
    assert torch.all(sig[1:] < sig[:-1])
    u = torch.rand(100)
    torch.testing.assert_close(unshift_sigma(shift_sigma(u, 3.0), 3.0), u)


def test_training_sigmas_threshold_fraction():
    s = sample_training_sigmas(20000, shift=3.0, generator=torch.Generator().manual_seed(0))
    frac = (s > 0.85).float().mean().item()
    # shifted uniform with shift 3: P(sigma > 0.85) = 1 - 0.85 / (3 - 2 * 0.85) ~= 0.346
    assert abs(frac - 0.346) < 0.02


def test_spatial_map_peak_and_base():
    cfg = DSCFGConfig(omega_peak=4.0, omega_base=1.0, sigma_rel=0.2)
    g = spatial_map(torch.tensor([[8.5 / 32, 16.5 / 32]]), torch.tensor([0.3]), (32, 32), cfg)
    assert g.shape == (1, 32, 32)
    iy, ix = divmod(int(g[0].argmax()), 32)
    assert abs(ix - 8) <= 1 and abs(iy - 16) <= 1
    assert math.isclose(g.max().item(), 1.0, abs_tol=0.02)
    assert math.isclose(g[0, 0, -1].item(), 0.25, abs_tol=1e-3)  # far away -> omega_base / omega_peak


def test_guidance_scale_target_only_and_floor():
    cfg = DSCFGConfig(omega_peak=4.0, omega_base=1.0, gamma=1.5, min_scale=1.0)
    g = torch.full((1, 4, 4), 0.25)
    mask = torch.zeros(1, 4, 4)
    mask[:, :2] = 1
    s_early = guidance_scale_map(0.9, g, mask, cfg)
    s_late = guidance_scale_map(0.01, g, mask, cfg)
    assert torch.all(s_early[:, 2:] == 0)  # outside target: unconditional
    assert torch.all(s_late[:, :2] == 1.0)  # floor inside the target
    no_floor = guidance_scale_map(0.01, g, None, DSCFGConfig(min_scale=0.0))
    assert no_floor.max() < 0.01
    vu, vc = torch.zeros(1, 2, 1, 4, 4), torch.ones(1, 2, 1, 4, 4)
    torch.testing.assert_close(apply_guidance(vu, vc, s_early)[0, 0], s_early)


def test_region_lock_pins_outside():
    g = torch.Generator().manual_seed(0)
    z, n, x = (torch.randn(1, 4, 2, 6, 6, generator=g) for _ in range(3))
    mask = torch.zeros(2, 6, 6)
    mask[:, 2:4, 2:4] = 1
    lock = RegionLock(z, n, mask, release_steps=1, release_dilate=1)
    y = lock(x, 0.5, step_idx=0, num_steps=10)
    ref = add_noise(z, n, 0.5)
    torch.testing.assert_close(y[..., 0, 0], ref[..., 0, 0])
    torch.testing.assert_close(y[..., 2, 2], x[..., 2, 2])
    y_last = lock(x, 0.0, step_idx=9, num_steps=10)
    torch.testing.assert_close(y_last[..., 1, 1], x[..., 1, 1])  # released ring
    torch.testing.assert_close(y_last[..., 0, 0], z[..., 0, 0])  # sigma 0 -> exactly source
    assert dilate_mask(mask, 1)[0].sum() == 16
