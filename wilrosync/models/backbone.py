"""Frozen Wan components: 3D VAE (with latent normalisation) and the umT5 text encoder."""

from __future__ import annotations

import html
import re

import torch
import torch.nn as nn

TEXT_MAX_LEN = 512


class WanVAE(nn.Module):
    """Wraps diffusers' ``AutoencoderKLWan`` and applies Wan's per-channel latent normalisation.

    encode: frames [B, 3, T, H, W] in [-1, 1] -> normalised latents [B, C, F, h, w]
    decode: normalised latents -> frames in [-1, 1]
    """

    def __init__(self, vae: nn.Module, sample: bool = False) -> None:
        super().__init__()
        self.vae = vae.eval()
        self.vae.requires_grad_(False)
        c = vae.config
        self.register_buffer("mean", torch.tensor(c.latents_mean).view(1, c.z_dim, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(c.latents_std).view(1, c.z_dim, 1, 1, 1), persistent=False)
        self.sample = sample

    @classmethod
    def from_pretrained(cls, repo: str, torch_dtype=torch.float32, **kw) -> WanVAE:
        from diffusers import AutoencoderKLWan

        return cls(AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch_dtype, **kw))

    @property
    def z_dim(self) -> int:
        return self.vae.config.z_dim

    @property
    def temporal_factor(self) -> int:
        return getattr(self.vae.config, "scale_factor_temporal", 4) or 4

    @property
    def spatial_factor(self) -> int:
        return getattr(self.vae.config, "scale_factor_spatial", 8) or 8

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        dist = self.vae.encode(frames.to(self.dtype)).latent_dist
        z = dist.sample() if self.sample else dist.mode()
        return ((z.float() - self.mean) / self.std).to(frames.dtype if frames.is_floating_point() else z.dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float() * self.std + self.mean
        return self.vae.decode(z.to(self.dtype), return_dict=False)[0].float().clamp(-1, 1)


def _clean_prompt(text: str) -> str:
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


class T5TextEncoder(nn.Module):
    """umT5 encoder, embeddings identical to diffusers' ``WanPipeline._get_t5_prompt_embeds``."""

    def __init__(self, tokenizer, encoder: nn.Module, max_len: int = TEXT_MAX_LEN) -> None:
        super().__init__()
        self.tokenizer, self.encoder, self.max_len = tokenizer, encoder.eval(), max_len
        self.encoder.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, repo: str, torch_dtype=torch.bfloat16, device="cpu") -> T5TextEncoder:
        from transformers import AutoTokenizer, UMT5EncoderModel

        tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
        kw = {"device_map": str(device)} if str(device).startswith("cuda") else {"low_cpu_mem_usage": True}
        enc = UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch_dtype, **kw)
        enc = enc.to(device)
        return cls(tok, enc)

    @torch.no_grad()
    def forward(self, prompts: str | list[str], trim: bool = False) -> torch.Tensor | list[torch.Tensor]:
        """Returns [B, max_len, D] zero-padded embeddings, or a list of trimmed [L_i, D] if ``trim``."""
        prompts = [prompts] if isinstance(prompts, str) else prompts
        prompts = [_clean_prompt(p) for p in prompts]
        dev = next(self.encoder.parameters()).device
        tok = self.tokenizer(prompts, padding="max_length", max_length=self.max_len, truncation=True,
                             add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        lens = tok.attention_mask.gt(0).sum(1)
        emb = self.encoder(tok.input_ids.to(dev), tok.attention_mask.to(dev)).last_hidden_state
        trimmed = [e[:n] for e, n in zip(emb, lens)]
        if trim:
            return trimmed
        return pad_text(trimmed, self.max_len)


def pad_text(embs: list[torch.Tensor], max_len: int = TEXT_MAX_LEN) -> torch.Tensor:
    """Zero-pad trimmed text embeddings to ``max_len`` (Wan attends over the zero padding)."""
    out = []
    for e in embs:
        e = e[:max_len]
        out.append(torch.cat([e, e.new_zeros(max_len - e.shape[0], e.shape[1])]))
    return torch.stack(out)
