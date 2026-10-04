"""End-to-end inference: source video + target audio (+ target face) -> lip-synced video.

Per window: VAE-encode the source -> progressive noise init (sigma = tau) -> denoise with
target-aware DS-CFG and the region lock -> decode -> blend overlapping windows -> composite into
the original-resolution frames through the feathered target mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..flow import add_noise, euler_step, inference_sigmas, sigma_to_timestep
from ..io.media import AUDIO_SR, pingpong_indices
from ..models.audio import frame_audio_windows
from .composite import composite
from .dscfg import DSCFGConfig, apply_guidance, guidance_scale_map, spatial_map
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
    max_side: int = 832
    size_multiple: int = 16
    region_lock: bool = True
    release_steps: int = 3
    release_dilate: int = 1
    feather_px: float = 6.0
    color_match: bool = True
    prompt: str = DEFAULT_PROMPT
    seed: int = 0
    loop_video: bool = True  # ping-pong the video when the audio is longer


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

    @classmethod
    def from_pretrained(cls, checkpoint: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                        vae_dtype: torch.dtype = torch.float32, text_device: str | None = None,
                        keep_text_encoder: bool = False) -> LipSyncPipeline:
        """``keep_text_encoder=False`` loads umT5 (~11 GB) only to encode the prompt, then frees it."""
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
            return cls(transformer, vae, load_text(), audio, device, dtype)
        return cls(transformer, vae, None, audio, device, dtype, text_loader=load_text)

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
                        full_frame: bool, cfg: InferenceConfig, generator) -> torch.Tensor:
        dev, dt = self.device, self.dtype
        noise = torch.randn(z_src.shape, generator=generator, dtype=torch.float32).to(dev, dt)
        sigmas = inference_sigmas(cfg.steps, cfg.tau, cfg.shift).tolist()
        x = add_noise(z_src, noise, sigmas[0])
        latent_hw = z_src.shape[-2:]
        g_map = spatial_map(mouth_lat.to(dev), fw_lat.to(dev), latent_hw, cfg.dscfg)
        tmask = None if full_frame else mask_lat.to(dev)
        lock = None
        if cfg.region_lock and not full_frame:
            lock = RegionLock(z_src, noise, mask_lat.to(dev), cfg.release_steps, cfg.release_dilate)
        text2 = torch.cat([text, text])
        zc2 = torch.cat([z_src, z_src])
        audio2 = torch.cat([audio_tokens, null_tokens])
        for i in range(cfg.steps):
            s, s1 = sigmas[i], sigmas[i + 1]
            t = sigma_to_timestep(s).to(dev).expand(2)
            v_c, v_u = self.transformer(torch.cat([x, x]), zc2, t, text2, audio2).chunk(2)
            scale = guidance_scale_map(s, g_map, tmask, cfg.dscfg, cfg.tau)
            v = apply_guidance(v_u.float(), v_c.float(), scale)
            x = euler_step(x, v, s, s1)
            if lock is not None:
                x = lock(x, s1, i, cfg.steps)
        return x

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

        windows = plan_windows(n, cfg.window, cfg.overlap, target.presence)
        weights = blend_weights(windows, n, cfg.overlap)
        out = frames.copy()
        acc: dict[int, torch.Tensor] = {}
        gen = torch.Generator().manual_seed(cfg.seed)

        for k, (win, wts) in enumerate(tqdm(list(zip(windows, weights)), disable=not progress, desc="windows")):
            decoded = self._run_window(frames, win, target, win_feats, text, (ph, pw), cfg, gen)
            for j in range(win.real_len):
                f = win.start + j
                acc[f] = acc.get(f, 0) + wts[j] * decoded[j]
            next_start = windows[k + 1].start if k + 1 < len(windows) else n
            ready = sorted(f for f in acc if f < next_start)
            if ready:
                self._flush(out, frames, target, acc, ready, cfg)
        return out

    def _run_window(self, frames, win: Window, target: TargetSignals, win_feats, text, proc_hw, cfg, gen):
        ph, pw = proc_hw
        pad_idx = np.concatenate([np.arange(win.start, win.end), np.full(win.length - win.real_len, win.end - 1)])
        src = _to_tensor(frames[pad_idx])
        src = F.interpolate(src, size=(ph, pw), mode="bilinear", align_corners=False, antialias=True)
        src = src.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # [1, 3, L, H', W']
        z_src = self.vae.encode(src).to(self.dtype)
        tf = self.vae.temporal_factor
        tsig = target.slice(win.start, win.end).pad_to(win.length)
        tsig_p = TargetSignals(tsig.resized_mask(ph, pw), tsig.mouth, tsig.face_width, tsig.presence,
                               tsig.full_frame)
        mask_lat, mouth_lat, fw_lat = tsig_p.to_latent(tuple(z_src.shape[-2:]), tf)
        a = win_feats[torch.as_tensor(pad_idx)].unsqueeze(0).to(self.device)
        tokens = self.transformer.encode_audio(a)
        null = self.transformer.null_audio(1, tokens.shape[1]).to(tokens.dtype)
        x = self._denoise_window(z_src, text, tokens, null, mask_lat, mouth_lat, fw_lat, tsig.full_frame, cfg, gen)
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
        res = composite(gen, src, mask, cfg.feather_px, cfg.color_match and not target.full_frame)
        out[ready] = _to_uint8(res)
