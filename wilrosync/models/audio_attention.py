"""Frame-local audio cross-attention inserted into every Wan DiT block.

The video tokens of latent frame f attend only to the audio tokens of latent frame f, which keeps
lip motion locked to the right moment in the audio. The output projection is zero-initialised so a
freshly added layer leaves the pretrained Wan block unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim, self.heads = dim, num_heads
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)
        self.to_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, hidden: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """hidden: [B, F*S, D] video tokens (frame-major order); audio: [B, F, N, D]. Returns the
        residual update (same shape as hidden)."""
        b, n_tok, d = hidden.shape
        f = audio.shape[1]
        if n_tok % f:
            raise ValueError(f"{n_tok} video tokens are not divisible by {f} latent frames")
        s = n_tok // f
        hd = d // self.heads
        w_dtype = self.to_q.weight.dtype
        # norm in fp32 (with fp32-cast weights) regardless of param/hidden dtype, then compute in w_dtype
        x = F.layer_norm(
            hidden.to(w_dtype).float(), self.norm.normalized_shape,
            self.norm.weight.float(), self.norm.bias.float(), self.norm.eps
        ).to(w_dtype)
        q = self.norm_q(self.to_q(x)).view(b * f, s, self.heads, hd).transpose(1, 2)
        a = audio.to(w_dtype).reshape(b * f, -1, d)
        k = self.norm_k(self.to_k(a)).view(b * f, -1, self.heads, hd).transpose(1, 2)
        v = self.to_v(a).view(b * f, -1, self.heads, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n_tok, d)
        return self.to_out(out)
