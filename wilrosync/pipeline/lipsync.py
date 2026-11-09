"""End-to-end inference: source video + target audio (+ target face) -> lip-synced video.

Per window: VAE-encode the source -> progressive noise init (sigma = tau) -> denoise with
target-aware DS-CFG and the region lock -> decode -> blend overlapping windows -> composite into
the original-resolution frames through the feathered target mask.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..flow import add_noise, euler_step, inference_sigmas, sigma_to_timestep
from ..io.media import AUDIO_SR, pingpong_indices
from ..models.audio import frame_audio_windows
from .composite import composite
from .dscfg import DSCFGConfig, apply_guidance, guidance_scale_map, spatial_map, temporal_weight
from .planner import Window, blend_weights, plan_windows
from .region_lock import RegionLock
from .target import TargetSignals

DEFAULT_PROMPT = "A person speaking clearly with natural, well-articulated lip and teeth movements."


@dataclass
class InferenceConfig:
    steps: int = 50
    tau: float = 0.92
    shift: float = 3.0
    dscfg: DSCFGConfig = field(default_factory=DSCFGConfig)
    window: int = 81
    overlap: int = 12
    # window chaining: each window after the first starts with the last ``chain`` frames already generated,
    # held fixed (region-locked) as context, so it continues them instead of starting cold. 1 + 4k (e.g. 9);
    # 0 = off. Replaces ``overlap`` (no cross-fade, so no doubled mouth at the cut).
    chain: int = 0
    # every window starts from the same noise (seeded by ``seed``) instead of fresh noise per window; tends to keep
    # fine detail (teeth, lip texture) more alike from window to window
    same_noise: bool = False
    max_side: int = 832
    size_multiple: int = 16
    region_lock: bool = True
    release_steps: int = 3
    release_dilate: int = 1
    feather_px: float = 6.0
    color_match: bool = True
    composite: bool = True  # False: return the decoded crops as generated (the caller blends them, e.g. face crops)
    prompt: str = DEFAULT_PROMPT
    seed: int = 0
    loop_video: bool = True  # ping-pong the video when the audio is longer
    # speed: drop the no-audio pass in steps where it cannot change the target region (guidance scale is at
    # its minimum everywhere in the mask; exact inside the mask, the region lock covers the rest)
    skip_uncond: bool = True
    cfg_stop_sigma: float = 0.0  # also drop it below this noise level (approximate; 0 = only where exact)
    profile: bool = False  # collect per-stage timings in LipSyncPipeline.timings


def load_infer_config(path: str | None = None) -> InferenceConfig:
    from omegaconf import OmegaConf

    cfg = OmegaConf.structured(InferenceConfig)
    if path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    return OmegaConf.to_object(cfg)


def processing_size(h: int, w: int, max_side: int, multiple: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(h, w))
    ph = max(multiple, int(round(h * scale / multiple)) * multiple)
    pw = max(multiple, int(round(w * scale / multiple)) * multiple)
    return ph, pw


def _to_tensor(frames: np.ndarray) -> torch.Tensor:
    """uint8 [T, H, W, 3] -> float [T, 3, H, W] in [-1, 1]."""
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1.0


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


class LipSyncPipeline:
    def __init__(self, transformer, vae, text_encoder, audio_encoder, device="cuda", dtype=torch.bfloat16,
                 text_loader=None):
        """``text_encoder`` may be None if ``text_loader`` (zero-arg callable) is given; the encoder is
        then loaded only to encode a new prompt and freed right after."""
        self.transformer = transformer.eval()
        self.vae = vae
        self.text_encoder = text_encoder
        self._text_loader = text_loader
        self._text_cache: dict[str, torch.Tensor] = {}
        self.audio_encoder = audio_encoder
        self.device = torch.device(device)
        self.dtype = dtype
        self.timings: dict[str, float] = {}
        self._profile = False

    @contextmanager
    def _timed(self, name: str):
        """Accumulate wall time of a stage in ``self.timings`` (only with ``cfg.profile``)."""
        if not self._profile:
            yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.time()
        yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timings[name] = self.timings.get(name, 0.0) + time.time() - t0

    def merge_lora(self) -> int:
        """Fold LoRA adapters into the base weights (same output, faster). Returns the number of merged layers."""
        try:
            from peft.tuners.lora import LoraLayer
        except ImportError:  # pragma: no cover
            return 0
        n = 0
        for m in self.transformer.modules():
            if isinstance(m, LoraLayer) and not m.merged:
                m.merge()
                n += 1
        return n

    @classmethod
    def from_pretrained(cls, checkpoint: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                        vae_dtype: torch.dtype = torch.float32, text_device: str | None = None,
                        keep_text_encoder: bool = False, merge_lora: bool = True) -> LipSyncPipeline:
        """``keep_text_encoder=False`` loads umT5 (~11 GB) only to encode the prompt, then frees it.
        ``merge_lora`` folds LoRA adapters into the weights for inference (same output, faster)."""
        from ..models.audio import WhisperAudioEncoder
        from ..models.backbone import T5TextEncoder, WanVAE
        from ..models.dit_lipsync import WanLipSyncTransformer

        transformer = WanLipSyncTransformer.from_checkpoint(checkpoint, torch_dtype=dtype).to(device)
        repo = transformer.cfg.base_repo
        vae = WanVAE.from_pretrained(repo, torch_dtype=vae_dtype).to(device)
        def load_text():
            return T5TextEncoder.from_pretrained(repo, torch_dtype=dtype, device=text_device or device)

        audio = WhisperAudioEncoder(transformer.cfg.audio_model, device=device)
        if keep_text_encoder:
            pipe = cls(transformer, vae, load_text(), audio, device, dtype)
        else:
            pipe = cls(transformer, vae, None, audio, device, dtype, text_loader=load_text)
        if merge_lora:
            pipe.merge_lora()
        return pipe

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        if prompt not in self._text_cache:
            enc = self.text_encoder if self.text_encoder is not None else self._text_loader()
            self._text_cache[prompt] = enc([prompt]).detach().cpu()
            if self.text_encoder is None:  # loaded on demand -> free it again
                del enc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return self._text_cache[prompt].to(self.device, self.dtype)

    # ------------------------------------------------------------------ helpers
    def _denoise_window(self, z_src, text, audio_tokens, null_tokens, mask_lat, mouth_lat, fw_lat,
                        full_frame: bool, cfg: InferenceConfig, generator, force_lock: bool = False) -> torch.Tensor:
        dev, dt = self.device, self.dtype
        if cfg.same_noise:
            generator = torch.Generator().manual_seed(cfg.seed)
        noise = torch.randn(z_src.shape, generator=generator, dtype=torch.float32).to(dev, dt)
        sigmas = inference_sigmas(cfg.steps, cfg.tau, cfg.shift).tolist()
        x = add_noise(z_src, noise, sigmas[0])
        latent_hw = z_src.shape[-2:]
        g_map = spatial_map(mouth_lat.to(dev), fw_lat.to(dev), latent_hw, cfg.dscfg)
        tmask = None if full_frame else mask_lat.to(dev)
        lock = None
        if (cfg.region_lock and not full_frame) or force_lock:  # force: chained context, also in full-frame mode
            lock = RegionLock(z_src, noise, mask_lat.to(dev), cfg.release_steps, cfg.release_dilate)
        text2 = torch.cat([text, text])
        zc2 = torch.cat([z_src, z_src])
        audio2 = torch.cat([audio_tokens, null_tokens])
        d = cfg.dscfg
        for i in range(cfg.steps):
            s, s1 = sigmas[i], sigmas[i + 1]
            if self._cond_only(s, cfg, lock is not None):
                t = sigma_to_timestep(s).to(dev).expand(1)
                v = self.transformer(x, z_src, t, text, audio_tokens).float()
            else:
                t = sigma_to_timestep(s).to(dev).expand(2)
                v_c, v_u = self.transformer(torch.cat([x, x]), zc2, t, text2, audio2).chunk(2)
                scale = guidance_scale_map(s, g_map, tmask, d, cfg.tau)
                v = apply_guidance(v_u.float(), v_c.float(), scale)
            x = euler_step(x, v, s, s1)
            if lock is not None:
                x = lock(x, s1, i, cfg.steps)
        return x

    @staticmethod
    def _cond_only(sigma: float, cfg: InferenceConfig, locked: bool) -> bool:
        """True when the no-audio pass can be skipped at this step.

        Inside the target the scale is max(omega_peak * w(sigma) * G, min_scale) with G <= 1, so once
        omega_peak * w(sigma) <= min_scale the guided velocity equals the conditional one there. Outside the
        target (scale 0 = unconditional) the region lock replaces the latent, so only its soft rim differs."""
        if not (cfg.skip_uncond and locked):
            return False
        d = cfg.dscfg
        peak = d.omega_peak * temporal_weight(sigma, d, cfg.tau) if d.enabled else d.static_scale
        return peak <= d.min_scale or sigma < cfg.cfg_stop_sigma

    # ------------------------------------------------------------------ main entry
    @torch.no_grad()
    def __call__(
        self,
        frames: np.ndarray,
        fps: float,
        wav: np.ndarray,
        target: TargetSignals | None = None,
        cfg: InferenceConfig | None = None,
        progress: bool = True,
    ) -> np.ndarray:
        """frames: uint8 [T, H, W, 3]; wav: mono float32 at 16 kHz. Returns uint8 frames whose length
        matches the audio (video ping-ponged if needed, or trimmed)."""
        cfg = cfg or InferenceConfig()
        self._profile = cfg.profile
        if cfg.profile:
            self.timings = {}
        t_start = time.time()
        t_src, h, w, _ = frames.shape
        n_audio = int(math.floor(len(wav) / AUDIO_SR * fps))
        n = n_audio if cfg.loop_video else min(t_src, n_audio)
        if n < 1:
            raise ValueError("audio is shorter than one video frame")
        idx = pingpong_indices(t_src, n)
        frames = frames[idx]
        target = (target or TargetSignals.whole_frame(t_src, h, w)).select(idx)
        ph, pw = processing_size(h, w, cfg.max_side, cfg.size_multiple)

        feats = self.audio_encoder(wav)
        win_feats = frame_audio_windows(feats, n, fps, window=self.transformer.cfg.audio_window)
        text = self.encode_prompt(cfg.prompt)

        out = frames.copy()
        gen = torch.Generator().manual_seed(cfg.seed)
        if cfg.chain:
            self._run_chained(out, frames, target, win_feats, text, (ph, pw), cfg, gen, n, progress)
            if cfg.profile:
                self.timings["total"] = time.time() - t_start
            return out
        windows = plan_windows(n, cfg.window, cfg.overlap, target.presence)
        weights = blend_weights(windows, n, cfg.overlap)
        acc: dict[int, torch.Tensor] = {}

        for k, (win, wts) in enumerate(tqdm(list(zip(windows, weights)), disable=not progress, desc="windows")):
            decoded = self._run_window(frames, win, target, win_feats, text, (ph, pw), cfg, gen)
            for j in range(win.real_len):
                f = win.start + j
                acc[f] = acc.get(f, 0) + wts[j] * decoded[j]
            next_start = windows[k + 1].start if k + 1 < len(windows) else n
            ready = sorted(f for f in acc if f < next_start)
            if ready:
                self._flush(out, frames, target, acc, ready, cfg)
        if cfg.profile:
            self.timings["total"] = time.time() - t_start
        return out

    def _run_chained(self, out, frames, target: TargetSignals, win_feats, text, proc_hw, cfg, gen, n, progress):
        """Windows overlapping by ``cfg.chain`` frames; the overlap is generated once (by the earlier window) and
        given to the next window as fixed context: its latent frames are region-locked to the generated frames,
        and the next window's output is used from the first frame after them."""
        tf = self.vae.temporal_factor
        if (cfg.chain - 1) % tf or cfg.chain < 1:
            raise ValueError(f"chain must be 1 + {tf}k, got {cfg.chain}")
        c = min(cfg.chain, cfg.window - tf)  # short windows: at least one new latent frame per window
        if cfg.overlap:
            raise ValueError("use either chain or overlap, not both")
        windows = plan_windows(n, cfg.window, c, target.presence)
        px: dict[int, torch.Tensor] = {}  # final decoded frames (processing size) not yet written
        prev = None
        for k, win in enumerate(tqdm(windows, disable=not progress, desc="windows")):
            ctx = prev.end - win.start if prev is not None and prev.end > win.start else 0
            n_lock = 1 + (ctx - 1) // tf if ctx else 0  # latent frames fully covered by the context
            n_ctx = 1 + tf * (n_lock - 1) if n_lock else 0  # their frames
            ctx_px = torch.stack([px[f] for f in range(win.start, win.start + n_ctx)]) if n_ctx else None
            decoded = self._run_window(frames, win, target, win_feats, text, proc_hw, cfg, gen,
                                       ctx_px=ctx_px, n_lock=n_lock)
            for j in range(n_ctx, win.real_len):
                px[win.start + j] = decoded[j]
            next_start = windows[k + 1].start if k + 1 < len(windows) else n
            ready = sorted(f for f in px if f < next_start)
            if ready:
                self._flush(out, frames, target, {f: px.pop(f) for f in ready}, ready, cfg)
            prev = win

    def _run_window(self, frames, win: Window, target: TargetSignals, win_feats, text, proc_hw, cfg, gen,
                    ctx_px: torch.Tensor | None = None, n_lock: int = 0):
        """``ctx_px`` [c, 3, H', W'] (-1..1): already generated frames that replace the first c source frames;
        the first ``n_lock`` latent frames (which cover them) are then held fixed."""
        ph, pw = proc_hw
        pad_idx = np.concatenate([np.arange(win.start, win.end), np.full(win.length - win.real_len, win.end - 1)])
        src = _to_tensor(frames[pad_idx])
        src = F.interpolate(src, size=(ph, pw), mode="bilinear", align_corners=False, antialias=True)
        if ctx_px is not None:
            src[: len(ctx_px)] = ctx_px.to(src.dtype)
        src = src.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # [1, 3, L, H', W']
        with self._timed("vae encode"):
            z_src = self.vae.encode(src).to(self.dtype)
        tf = self.vae.temporal_factor
        tsig = target.slice(win.start, win.end).pad_to(win.length)
        tsig_p = TargetSignals(tsig.resized_mask(ph, pw), tsig.mouth, tsig.face_width, tsig.presence,
                               tsig.full_frame)
        mask_lat, mouth_lat, fw_lat = tsig_p.to_latent(tuple(z_src.shape[-2:]), tf)
        if n_lock:
            mask_lat = mask_lat.clone()
            mask_lat[:n_lock] = 0  # context: locked to the frames generated by the previous window
        a = win_feats[torch.as_tensor(pad_idx)].unsqueeze(0).to(self.device)
        tokens = self.transformer.encode_audio(a)
        null = self.transformer.null_audio(1, tokens.shape[1]).to(tokens.dtype)
        with self._timed("denoise"):
            x = self._denoise_window(z_src, text, tokens, null, mask_lat, mouth_lat, fw_lat, tsig.full_frame,
                                     cfg, gen, force_lock=n_lock > 0)
        with self._timed("vae decode"):
            video = self.vae.decode(x)[0].permute(1, 0, 2, 3)  # [L, 3, H', W']
        return video[: win.real_len].float().cpu()

    def _flush(self, out, frames, target: TargetSignals, acc, ready, cfg: InferenceConfig):
        h, w = frames.shape[1:3]
        gen = torch.stack([acc.pop(f) for f in ready])
        gen = F.interpolate(gen, size=(h, w), mode="bicubic", align_corners=False).clamp(-1, 1)
        src = _to_tensor(frames[ready])
        mask = target.mask[ready].float()
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask[:, None], size=(h, w), mode="bilinear")[:, 0]
        if not cfg.composite:
            out[ready] = _to_uint8(gen)
            return
        res = composite(gen, src, mask, cfg.feather_px, cfg.color_match and not target.full_frame)
        out[ready] = _to_uint8(res)
