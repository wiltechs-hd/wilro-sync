"""Audio conditioning: Whisper features -> per-video-frame windows -> per-latent-frame tokens.

Shapes
------
whisper features : [T_a, L, C]           50 Hz, all L hidden layers of the (frozen) encoder
frame windows    : [T, W, L, C]          W audio features centred on each video frame
audio tokens     : [B, F_lat, 4 * W, D]  tokens attended by the frames of one latent frame
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

WHISPER_SR = 16000
WHISPER_FPS = 50.0  # encoder output rate
WHISPER_CHUNK_S = 30.0


def frame_audio_windows(
    feats: torch.Tensor, num_frames: int, fps: float, window: int = 10, offset_frames: float = 0.0
) -> torch.Tensor:
    """Gather ``window`` audio features centred on each video frame.

    feats: [T_a, L, C] at 50 Hz -> [num_frames, window, L, C] (indices clamped at the borders).
    ``offset_frames`` shifts audio relative to video (positive = audio later), for A/V sync fixes.
    """
    t_a = feats.shape[0]
    centers = (torch.arange(num_frames, dtype=torch.float64) + offset_frames) / fps * WHISPER_FPS
    starts = torch.round(centers).long() - window // 2
    idx = starts.view(-1, 1) + torch.arange(window).view(1, -1)
    idx = idx.clamp(0, max(t_a - 1, 0))
    return feats[idx]


class WhisperAudioEncoder(nn.Module):
    """Frozen Whisper encoder returning all hidden states at 50 Hz (any audio length)."""

    def __init__(self, model_name: str = "openai/whisper-tiny", device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.encoder = WhisperModel.from_pretrained(model_name, torch_dtype=dtype).get_encoder().to(device).eval()
        self.encoder.requires_grad_(False)
        self.device_, self.dtype_ = torch.device(device), dtype

    @property
    def num_layers(self) -> int:
        return self.encoder.config.encoder_layers + 1

    @property
    def dim(self) -> int:
        return self.encoder.config.d_model

    @torch.no_grad()
    def forward(self, wav: np.ndarray | torch.Tensor) -> torch.Tensor:
        """wav: mono float32 at 16 kHz -> [T_a, L, C] with T_a = ceil(duration * 50)."""
        wav = np.asarray(wav, dtype=np.float32)
        chunk = int(WHISPER_CHUNK_S * WHISPER_SR)
        n_out = max(1, math.ceil(len(wav) / WHISPER_SR * WHISPER_FPS))
        outs = []
        for s in range(0, max(len(wav), 1), chunk):
            piece = wav[s : s + chunk]
            inp = self.feature_extractor(piece, sampling_rate=WHISPER_SR, return_tensors="pt").input_features
            hs = self.encoder(inp.to(self.device_, self.dtype_), output_hidden_states=True).hidden_states
            h = torch.stack(hs, dim=1)[0]  # [L, 1500, C]
            valid = math.ceil(len(piece) / WHISPER_SR * WHISPER_FPS)
            outs.append(h[:, :valid].permute(1, 0, 2))
        return torch.cat(outs, 0)[:n_out].float()


class AudioProjector(nn.Module):
    """Turns per-frame Whisper windows into audio tokens for each latent frame.

    Latent frame 0 covers video frame 0 only; we left-pad with copies of frame 0 so every latent
    frame covers ``temporal_factor`` video frames, giving ``temporal_factor * window`` tokens each.
    """

    def __init__(self, num_layers: int, in_dim: int, dim: int, window: int = 10, temporal_factor: int = 4,
                 hidden: int | None = None) -> None:
        super().__init__()
        self.num_layers, self.in_dim, self.dim = num_layers, in_dim, dim
        self.window, self.temporal_factor = window, temporal_factor
        self.num_tokens = window * temporal_factor
        hidden = hidden or 2 * dim
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.in_proj = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, 1, self.num_tokens, dim))
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm_out = nn.LayerNorm(dim)
        self.null_tokens = nn.Parameter(torch.zeros(self.num_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.null_tokens, std=0.02)

    def num_latent_frames(self, num_frames: int) -> int:
        return (num_frames - 1) // self.temporal_factor + 1

    def forward(self, windows: torch.Tensor, drop: torch.Tensor | None = None) -> torch.Tensor:
        """windows: [B, T, W, L, C] (T = 1 + 4k) -> tokens [B, F_lat, 4W, D].

        ``drop``: optional bool [B]; dropped samples get the learned null tokens (for CFG).
        """
        b, t, w, num_l, c = windows.shape
        if (t - 1) % self.temporal_factor:
            raise ValueError(f"num frames must be 1 + {self.temporal_factor}k, got {t}")
        weights = torch.softmax(self.layer_logits, 0).to(windows.dtype)
        x = torch.einsum("btwlc,l->btwc", windows, weights)
        pad = x[:, :1].expand(b, self.temporal_factor - 1, w, c)
        x = torch.cat([pad, x], 1)  # [B, 4*F_lat, W, C]
        f_lat = x.shape[1] // self.temporal_factor
        x = x.reshape(b, f_lat, self.temporal_factor * w, c)
        x = self.in_proj(x) + self.pos
        x = self.norm_out(x + self.mlp(x))
        if drop is None:
            drop = torch.zeros(b, dtype=torch.bool, device=x.device)
        # always route through the null tokens so DDP sees them in every step's graph
        null = self.null_tokens.to(x.dtype).expand_as(x)
        return torch.where(drop.view(-1, 1, 1, 1), null, x)

    def null(self, batch: int, num_latent_frames: int) -> torch.Tensor:
        return self.null_tokens.expand(batch, num_latent_frames, *self.null_tokens.shape)
