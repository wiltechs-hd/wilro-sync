"""Wan DiT extended for lip sync (OmniSync Fig. 2).

Changes to the pretrained diffusers ``WanTransformer3DModel`` (weights are reused as-is):

1. **Condition input**: the patch embedding is widened so the source-video latent ``z_cond`` (and,
   for the target-aware variant, a mask channel) is channel-concatenated with the noisy latent.
   New input weights are zero-initialised.
2. **Audio cross-attention**: one frame-local ``AudioCrossAttention`` per block, inserted between
   the text cross-attention and the FFN. Output projections are zero-initialised.
3. **Audio projector**: Whisper windows -> audio tokens.

At initialisation the model therefore computes exactly what the base Wan model computes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .audio import AudioProjector
from .audio_attention import AudioCrossAttention

CONFIG_NAME = "wilrosync_config.json"
WEIGHTS_NAME = "wilrosync_trainable.safetensors"


@dataclass
class LipSyncModelConfig:
    base_repo: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    use_mask_channel: bool = False
    audio_layers: int = 5  # whisper-tiny: 4 encoder layers + embeddings
    audio_dim: int = 384  # whisper-tiny d_model
    audio_window: int = 10
    temporal_factor: int = 4
    audio_model: str = "openai/whisper-tiny"
    lora: dict | None = field(default=None)  # {"rank": 64, "alpha": 64, "targets": [...]} when LoRA is used


class WanLipSyncTransformer(nn.Module):
    def __init__(self, base: nn.Module, cfg: LipSyncModelConfig) -> None:
        super().__init__()
        self.base = base
        self.cfg = cfg
        bc = base.config
        self.latent_channels = bc.out_channels or bc.in_channels
        self.inner_dim = bc.num_attention_heads * bc.attention_head_dim
        self._expand_patch_embedding()
        self.audio_proj = AudioProjector(
            cfg.audio_layers, cfg.audio_dim, self.inner_dim, cfg.audio_window, cfg.temporal_factor
        )
        self.audio_attn = nn.ModuleList(
            [AudioCrossAttention(self.inner_dim, bc.num_attention_heads, eps=bc.eps) for _ in base.blocks]
        )
        # NB: some base params (AdaLN tables) are kept in fp32 by diffusers, so we can't use
        # next(base.parameters()).dtype -- the patch embedding carries the compute dtype.
        dtype = self.base.patch_embedding.weight.dtype
        self.audio_proj.to(dtype)
        self.audio_attn.to(dtype)
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------ construction
    @property
    def num_input_channels(self) -> int:
        return 2 * self.latent_channels + (1 if self.cfg.use_mask_channel else 0)

    def _expand_patch_embedding(self) -> None:
        old: nn.Conv3d = self.base.patch_embedding
        new_in = self.num_input_channels
        if old.in_channels == new_in:
            return
        if old.in_channels != self.latent_channels:
            raise ValueError(f"unexpected patch embedding with {old.in_channels} input channels")
        new = nn.Conv3d(new_in, old.out_channels, old.kernel_size, old.stride, dtype=old.weight.dtype,
                        device=old.weight.device)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, : old.in_channels].copy_(old.weight)
            new.bias.copy_(old.bias)
        self.base.patch_embedding = new

    @classmethod
    def from_base(cls, cfg: LipSyncModelConfig, torch_dtype: torch.dtype = torch.bfloat16, **kwargs):
        from diffusers import WanTransformer3DModel

        base = WanTransformer3DModel.from_pretrained(cfg.base_repo, subfolder="transformer",
                                                     torch_dtype=torch_dtype, **kwargs)
        return cls(base, cfg)

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, torch_dtype: torch.dtype = torch.bfloat16, base: nn.Module | None = None):
        from safetensors.torch import load_file

        with open(os.path.join(ckpt_dir, CONFIG_NAME)) as f:
            cfg = LipSyncModelConfig(**json.load(f))
        if base is None:
            model = cls.from_base(cfg, torch_dtype)
        else:
            model = cls(base, cfg)
        if cfg.lora:
            from .lora import add_lora

            add_lora(model, **cfg.lora)
        state = load_file(os.path.join(ckpt_dir, WEIGHTS_NAME))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}...")
        # cast only the new / LoRA weights; diffusers keeps norms, time embedder and AdaLN tables in fp32
        model.audio_proj.to(torch_dtype)
        model.audio_attn.to(torch_dtype)
        model.base.patch_embedding.to(torch_dtype)
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.data = p.data.to(torch_dtype)
        return model

    def save_checkpoint(self, ckpt_dir: str) -> None:
        from safetensors.torch import save_file

        os.makedirs(ckpt_dir, exist_ok=True)
        state = {k: v.detach().contiguous().cpu() for k, v in self.trainable_state_dict().items()}
        save_file(state, os.path.join(ckpt_dir, WEIGHTS_NAME))
        with open(os.path.join(ckpt_dir, CONFIG_NAME), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

    # ------------------------------------------------------------------ trainability
    def set_trainable(self, mode: str = "adapter") -> None:
        """``adapter``: new layers + widened patch embedding (+ AdaLN tables).
        ``lora``: adapter + LoRA params (LoRA must be added first). ``full``: everything."""
        if mode not in {"adapter", "lora", "full"}:
            raise ValueError(mode)
        self.requires_grad_(mode == "full")
        if mode == "full":
            return
        for m in (self.audio_proj, self.audio_attn, self.base.patch_embedding):
            m.requires_grad_(True)
        for name, p in self.base.named_parameters():
            if name.endswith("scale_shift_table"):
                p.requires_grad_(True)
            if mode == "lora" and "lora_" in name:
                p.requires_grad_(True)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        sd = self.state_dict()
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        # always keep the new modules even if frozen at save time
        names |= {n for n in sd if n.startswith(("audio_proj.", "audio_attn.", "base.patch_embedding."))}
        names |= {n for n in sd if "lora_" in n}
        return {n: sd[n] for n in sorted(names)}

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    # ------------------------------------------------------------------ forward
    def encode_audio(self, windows: torch.Tensor, drop: torch.Tensor | None = None) -> torch.Tensor:
        return self.audio_proj(windows.to(self.audio_proj.in_proj.weight.dtype), drop)

    def null_audio(self, batch: int, num_latent_frames: int) -> torch.Tensor:
        return self.audio_proj.null(batch, num_latent_frames)

    @staticmethod
    def _block_forward(block, audio_attn, hidden, encoder_hidden, temb, rotary, audio_tokens):
        """diffusers ``WanTransformerBlock.forward`` with audio cross-attention after text cross-attn."""
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (
                block.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa, scale_msa, gate_msa = shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2)
            c_shift, c_scale, c_gate = c_shift.squeeze(2), c_scale.squeeze(2), c_gate.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (
                block.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. self-attention
        norm = (block.norm1(hidden.float()) * (1 + scale_msa) + shift_msa).type_as(hidden)
        attn = block.attn1(norm, None, None, rotary)
        hidden = (hidden.float() + attn * gate_msa).type_as(hidden)
        # 2. text cross-attention
        norm = block.norm2(hidden.float()).type_as(hidden)
        hidden = hidden + block.attn2(norm, encoder_hidden, None, None)
        # 3. audio cross-attention (new)
        if audio_tokens is not None:
            hidden = hidden + audio_attn(hidden, audio_tokens)
        # 4. feed-forward
        norm = (block.norm3(hidden.float()) * (1 + c_scale) + c_shift).type_as(hidden)
        ff = block.ffn(norm)
        return (hidden.float() + ff.float() * c_gate).type_as(hidden)

    def forward(
        self,
        x_t: torch.Tensor,
        z_cond: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: torch.Tensor,
        audio_tokens: torch.Tensor | None,
        mask_lat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_t, z_cond: [B, C, F, H, W]; timestep: [B] (sigma * 1000) or [B, seq];
        text_emb: [B, L, text_dim]; audio_tokens: [B, F, N, D] or None (no audio path at all).
        Returns the predicted velocity [B, C, F, H, W]."""
        base = self.base
        b, _, f, h, w = x_t.shape
        p_t, p_h, p_w = base.config.patch_size
        pf, ph, pw = f // p_t, h // p_h, w // p_w

        inputs = [x_t, z_cond]
        if self.cfg.use_mask_channel:
            inputs.append(mask_lat if mask_lat is not None else torch.ones_like(x_t[:, :1]))
        dtype = base.patch_embedding.weight.dtype
        hs = torch.cat([t.to(dtype) for t in inputs], 1)
        rotary = base.rope(hs)
        hs = base.patch_embedding(hs).flatten(2).transpose(1, 2).contiguous()

        timestep = torch.as_tensor(timestep, device=hs.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(b)
        ts_seq_len = None
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        temb, tproj, enc, _ = base.condition_embedder(timestep, text_emb.to(dtype), None, timestep_seq_len=ts_seq_len)
        tproj = tproj.unflatten(2, (6, -1)) if ts_seq_len is not None else tproj.unflatten(1, (6, -1))

        if audio_tokens is not None:
            audio_tokens = audio_tokens.to(dtype)
            if audio_tokens.shape[1] != pf:
                raise ValueError(f"audio tokens cover {audio_tokens.shape[1]} latent frames, video has {pf}")

        for i, block in enumerate(base.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hs = checkpoint(self._block_forward, block, self.audio_attn[i], hs, enc, tproj, rotary,
                                audio_tokens, use_reentrant=False)
            else:
                hs = self._block_forward(block, self.audio_attn[i], hs, enc, tproj, rotary, audio_tokens)

        if temb.ndim == 3:
            shift, scale = (base.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            shift, scale = (base.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hs = (base.norm_out(hs.float()) * (1 + scale) + shift).type_as(hs)
        hs = base.proj_out(hs)
        hs = hs.reshape(b, pf, ph, pw, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)
