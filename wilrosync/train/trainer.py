"""Training loop (accelerate): conditional flow matching with timestep-dependent sampling.

    x_sigma = (1 - sigma) z_ab + sigma eps,   target = eps - z_ab
    loss    = mean( W * || v_theta(x_sigma, z_cd, A_ab, text, sigma) - target ||^2 )
            (+ lpips_weight * LPIPS on the decoded mouth crop at low sigma, optional; see perceptual.py)

W = 1 + lambda * G_mouth optionally up-weights the mouth region. Audio / text are dropped at random
so the same network also learns the unconditional prediction used by DS-CFG.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from ..flow import add_noise, sigma_to_timestep, target_velocity
from ..models.dit_lipsync import WanLipSyncTransformer
from ..models.lora import add_lora
from ..pipeline.dscfg import DSCFGConfig, spatial_map
from .config import TrainConfig


def build_model(cfg: TrainConfig, base_dtype: torch.dtype) -> WanLipSyncTransformer:
    if cfg.tiny_base is not None:
        from diffusers import WanTransformer3DModel

        kw = dict(cfg.tiny_base)
        if "patch_size" in kw:
            kw["patch_size"] = tuple(kw["patch_size"])
        base = WanTransformer3DModel(**kw).to(base_dtype)
        model = WanLipSyncTransformer(base, cfg.model)
    else:
        model = WanLipSyncTransformer.from_base(cfg.model, torch_dtype=base_dtype)
    if cfg.train_mode == "lora":
        add_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
    model.set_trainable(cfg.train_mode)
    for p in model.parameters():  # fp32 master weights for everything we train
        if p.requires_grad:
            p.data = p.data.float()
    if cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    return model


def build_dataset(cfg: TrainConfig):
    from ..data.datasets import LatentClipDataset, SyntheticClipDataset, TimestepDependentSampler

    d = cfg.data
    if d.synthetic:
        c, f, h, w = d.synthetic_shape
        kw = dict(c=c, f=f, h=h, w=w, audio_window=cfg.model.audio_window, audio_layers=cfg.model.audio_layers,
                  audio_dim=cfg.model.audio_dim,
                  text_dim=(cfg.tiny_base or {}).get("text_dim", 4096))
        pairs, arb = SyntheticClipDataset(seed=1, **kw), SyntheticClipDataset(seed=2, **kw)
    else:
        if not d.arbitrary_index:
            raise ValueError("data.arbitrary_index is required")
        arb = LatentClipDataset.from_index(d.arbitrary_index, kind="arbitrary")
        pairs = LatentClipDataset.from_index(d.pairs_index, kind="pseudo_pair") if d.pairs_index else None
    return TimestepDependentSampler(pairs, arb, d.tds_threshold, d.sigma_mode, d.sigma_shift, d.tds_enabled,
                                    seed=cfg.seed)


def mouth_weight(mouth: torch.Tensor, face_w: torch.Tensor, latent_hw, lam: float, sigma_rel: float):
    """[B, 1, F, h, w] loss weights 1 + lam * Gaussian(mouth)."""
    cfg = DSCFGConfig(omega_peak=1.0, omega_base=0.0, sigma_rel=sigma_rel)
    g = torch.stack([spatial_map(m, fw, latent_hw, cfg) for m, fw in zip(mouth, face_w)])
    return (1.0 + lam * g).unsqueeze(1)


def _make_optimizer(cfg: TrainConfig, model: WanLipSyncTransformer):
    new, old = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_new = n.startswith(("audio_proj.", "audio_attn.", "base.patch_embedding."))
        (new if is_new else old).append(p)
    groups = [g for g in ({"params": new, "lr": cfg.lr_new}, {"params": old, "lr": cfg.lr}) if g["params"]]
    if cfg.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))


def _lr_lambda(cfg: TrainConfig):
    """Learning-rate factor per optimizer step: linear warmup, then constant, then (optionally) cosine decay from
    ``lr_decay_start`` down to ``lr_min_ratio`` at ``max_steps``."""
    def f(step: int) -> float:
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        start = cfg.lr_decay_start
        if start is None or step <= start or cfg.max_steps <= start:
            return 1.0
        p = min(1.0, (step - start) / (cfg.max_steps - start))
        return cfg.lr_min_ratio + (1.0 - cfg.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))

    return f


def _apply_config_lrs(cfg: TrainConfig, opt, sched) -> list[float]:
    """After resuming: the saved optimizer / scheduler state carries the OLD learning rates, so a changed ``lr`` or
    ``lr_new`` would be ignored. Reset the base rates from the config (param-group order as in _make_optimizer)."""
    inner_opt = getattr(opt, "optimizer", opt)
    inner_sched = getattr(sched, "scheduler", sched)
    # groups are [new layers, pretrained/LoRA]; with one group only the new layers train (adapter mode)
    bases = [cfg.lr_new, cfg.lr] if len(inner_opt.param_groups) == 2 else [cfg.lr_new]
    for g, base in zip(inner_opt.param_groups, bases):
        g["initial_lr"] = base
    inner_sched.base_lrs = list(bases)
    factor = inner_sched.lr_lambdas[0](inner_sched.last_epoch)
    for g, base in zip(inner_opt.param_groups, bases):
        g["lr"] = base * factor
    return bases


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def reset(self, model: torch.nn.Module) -> None:
        """Start averaging again from the model's current weights."""
        params = dict(model.named_parameters())
        for n in self.shadow:
            self.shadow[n] = params[n].detach().clone().float()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].lerp_(p.detach().float(), 1.0 - self.decay)

    @torch.no_grad()
    def swap(self, model: torch.nn.Module) -> None:
        params = dict(model.named_parameters())
        for n, s in self.shadow.items():
            tmp = params[n].detach().clone()
            params[n].copy_(s)
            self.shadow[n] = tmp.float()


def train(cfg: TrainConfig) -> dict:
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    acc = Accelerator(mixed_precision=cfg.mixed_precision, gradient_accumulation_steps=cfg.grad_accum)
    set_seed(cfg.seed)
    base_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32
    model = build_model(cfg, base_dtype)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    acc.print(f"trainable params: {n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M ({cfg.train_mode})")

    dataset = build_dataset(cfg)
    from ..data.datasets import collate

    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.data.num_workers, collate_fn=collate)
    opt = _make_optimizer(cfg, model)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg))
    model, opt, loader, sched = acc.prepare(model, opt, loader, sched)
    ema = EMA(acc.unwrap_model(model), cfg.ema_decay) if cfg.ema_decay > 0 else None

    null_text = None
    if cfg.data.null_text:
        from safetensors.torch import load_file

        null_text = load_file(cfg.data.null_text)["text"]

    step = 0
    if cfg.resume_from:
        acc.load_state(os.path.join(cfg.resume_from, "state"))
        with open(os.path.join(cfg.resume_from, "step.json")) as f:
            step = json.load(f)["step"]
        bases = _apply_config_lrs(cfg, opt, sched)
        if ema is not None:  # the EMA was built before the weights were loaded: restart it from the resumed weights
            ema.reset(acc.unwrap_model(model))
            acc.print("EMA restarted from the resumed weights")
        acc.print(f"resumed at step {step}; learning rates {bases} x {_lr_lambda(cfg)(step):.3f}"
                  + (f", cosine decay from step {cfg.lr_decay_start} to {cfg.lr_min_ratio:g}x at {cfg.max_steps}"
                     if cfg.lr_decay_start is not None else ""))
    if acc.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

    perc = _build_lpips(cfg, acc.device, base_dtype) if cfg.lpips_weight > 0 else None
    if perc is not None:
        acc.print(f"LPIPS loss: weight {cfg.lpips_weight}, sigma < {cfg.lpips_max_sigma}, "
                  f"{cfg.lpips_latent_frames} latent frames, crop {cfg.lpips_crop} cells, every {cfg.lpips_every}")

    model.train()
    history, it, micro = [], iter(loader), 0
    t_log, s_log = time.time(), step
    lp_hist: list[float] = []
    while step < cfg.max_steps:
        batch = next(it)
        with acc.accumulate(model):
            use_lp = perc is not None and micro % max(1, cfg.lpips_every) == 0
            micro += 1
            loss = _loss(cfg, acc.unwrap_model(model), model, batch, null_text, perc if use_lp else None)
            if use_lp and perc.last is not None:
                lp_hist.append(perc.last)
            acc.backward(loss)
            if acc.sync_gradients and cfg.max_grad_norm > 0:
                acc.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if not acc.sync_gradients:
            continue
        step += 1
        if ema is not None:
            ema.update(acc.unwrap_model(model))
        lv = acc.gather(loss.detach()).mean().item()
        history.append(lv)
        if step % cfg.log_every == 0 or step == 1:
            now = time.time()
            rate = (step - s_log) / max(now - t_log, 1e-6)  # since the last log line (not since step 0)
            t_log, s_log = now, step
            lr_now = sched.get_last_lr()[0]
            lp_mean = sum(lp_hist) / len(lp_hist) if lp_hist else None
            acc.print(f"step {step:>7d}  loss {lv:.4f}  lr {lr_now:.2e}  {1 / max(rate, 1e-9):.1f} s/step"
                      + (f"  lpips {lp_mean:.4f} (n={len(lp_hist)})" if lp_mean is not None else ""))
            if acc.is_main_process:
                window = history[-cfg.log_every:]
                rec = {"step": step, "loss": sum(window) / len(window), "lr": lr_now, "it_s": rate,
                       "time": time.time()}
                if lp_mean is not None:
                    rec["lpips"] = lp_mean
                with open(os.path.join(cfg.output_dir, "log.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            lp_hist = []
        if step % cfg.save_every == 0 or step == cfg.max_steps:
            _save(acc, model, ema, cfg, step)
    return {"steps": step, "losses": history}


def _loss(cfg: TrainConfig, core: WanLipSyncTransformer, model, batch: dict, null_text, perc=None) -> torch.Tensor:
    z_ab, z_cd, sigma = batch["z_ab"].float(), batch["z_cd"].float(), batch["sigma"].float()
    b = z_ab.shape[0]
    dev = z_ab.device
    noise = torch.randn_like(z_ab)
    x_t = add_noise(z_ab, noise, sigma)
    target = target_velocity(z_ab, noise)

    r = torch.rand(3, b, device=dev)
    both = r[0] < cfg.p_drop_both
    drop_a = both | (r[1] < cfg.p_drop_audio)
    drop_t = both | (r[2] < cfg.p_drop_text)
    text = batch["text"].float()
    if drop_t.any():
        if null_text is not None:
            from ..models.backbone import pad_text

            null = pad_text([null_text.to(text)], text.shape[1])
        else:
            null = torch.zeros_like(text[:1])
        text = torch.where(drop_t.view(-1, 1, 1), null.expand_as(text), text)

    tokens = core.encode_audio(batch["audio"].float(), drop=drop_a)
    v = model(x_t, z_cd, sigma_to_timestep(sigma).to(dev), text, tokens)
    err = (v.float() - target.float()) ** 2
    if cfg.mouth_loss_weight > 0:
        w = mouth_weight(batch["mouth"], batch["face_w"], z_ab.shape[-2:], cfg.mouth_loss_weight,
                         cfg.mouth_sigma_rel)
        err = err * w / w.mean()
    if not math.isfinite(err.mean().item()):
        raise FloatingPointError("non-finite loss")
    loss = err.mean()
    if perc is not None:  # LPIPS on the decoded mouth crop of the one-step guess (low-noise samples only)
        z0_hat = x_t.float() - sigma.view(-1, 1, 1, 1, 1) * v.float()
        lp = perc(z0_hat, z_ab, sigma, batch["mouth"])
        if lp is not None:
            loss = loss + cfg.lpips_weight * lp
    return loss


def _build_lpips(cfg: TrainConfig, device, dtype):
    """Frozen VAE + LPIPS network for ``lpips_weight > 0``."""
    from ..models.backbone import WanVAE
    from . import perceptual

    if cfg.tiny_base is not None:  # tests: a tiny random VAE with the tiny model's latent channels
        from diffusers import AutoencoderKLWan

        z = cfg.tiny_base.get("in_channels", 16)
        vae = WanVAE(AutoencoderKLWan(base_dim=16, z_dim=z, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
                                      latents_mean=[0.0] * z, latents_std=[1.0] * z))
    else:
        vae = WanVAE.from_pretrained(cfg.model.base_repo, torch_dtype=dtype)
    return perceptual.LatentLPIPS(vae.to(device), perceptual.make_lpips(cfg.lpips_net).to(device),
                                  crop=cfg.lpips_crop, latent_frames=cfg.lpips_latent_frames,
                                  max_sigma=cfg.lpips_max_sigma)


def _save(acc, model, ema, cfg: TrainConfig, step: int) -> None:
    acc.wait_for_everyone()
    ckpt = os.path.join(cfg.output_dir, f"step_{step:07d}")
    acc.save_state(os.path.join(ckpt, "state"))
    if acc.is_main_process:
        core = acc.unwrap_model(model)
        core.save_checkpoint(ckpt)
        if ema is not None:
            ema.swap(core)
            core.save_checkpoint(os.path.join(ckpt, "ema"))
            ema.swap(core)
        with open(os.path.join(ckpt, "step.json"), "w") as f:
            json.dump({"step": step}, f)
        if cfg.keep_last > 0:
            import shutil

            old = sorted(d for d in os.listdir(cfg.output_dir) if d.startswith("step_"))[: -cfg.keep_last]
            for d in old:
                shutil.rmtree(os.path.join(cfg.output_dir, d), ignore_errors=True)
