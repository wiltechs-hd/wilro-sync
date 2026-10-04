"""Training data: pre-computed latent clips + the timestep-dependent sampler (OmniSync Sec. 3.2).

A clip is one ``.safetensors`` file produced by ``scripts/prepare_clips.py`` with tensors

    z_cd   [C, F, h, w]   condition video latent (V_cd)
    z_ab   [C, F, h, w]   target video latent (V_ab)
    audio  [T, W, L, C_a] Whisper windows aligned to the T frames of V_ab (T = 1 + 4(F-1))
    text   [L_t, D_t]     trimmed umT5 embedding of the clip caption
    mouth  [F, 2]         mouth centre per latent frame (normalised), optional
    face_w [F]            face width per latent frame (fraction of width), optional

and an index ``.jsonl`` with one ``{"path": ..., "kind": "pseudo_pair" | "arbitrary"}`` per line.

* ``pseudo_pair`` (MEAD-style): V_cd and V_ab are different utterances of the same speaker filmed
  from the same camera, so pose matches but lips differ. Used for high-noise steps.
* ``arbitrary``: V_cd and V_ab are two segments of the same in-the-wild video. Used for the rest.
"""

from __future__ import annotations

import json
import os
import random

import torch
from torch.utils.data import Dataset, IterableDataset

from ..flow import sample_training_sigmas
from ..models.backbone import TEXT_MAX_LEN, pad_text


def read_index(path: str) -> list[dict]:
    root = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not os.path.isabs(row["path"]):
                row["path"] = os.path.join(root, row["path"])
            rows.append(row)
    return rows


class LatentClipDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    @classmethod
    def from_index(cls, path: str, kind: str | None = None) -> LatentClipDataset:
        rows = read_index(path)
        if kind:
            rows = [r for r in rows if r.get("kind") == kind]
        return cls(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        from safetensors.torch import load_file

        d = load_file(self.rows[i]["path"])
        f = d["z_ab"].shape[1]
        d.setdefault("mouth", torch.tensor([0.5, 0.62]).expand(f, 2).clone())
        d.setdefault("face_w", torch.full((f,), 0.45))
        return d


class SyntheticClipDataset(Dataset):
    """Random tensors with the right shapes, for smoke tests and benchmarking."""

    def __init__(self, n=8, c=16, f=3, h=8, w=8, audio_window=10, audio_layers=5, audio_dim=384, text_dim=4096,
                 text_len=12, seed=0) -> None:
        self.n, self.shape = n, (c, f, h, w)
        self.audio_shape = (1 + 4 * (f - 1), audio_window, audio_layers, audio_dim)
        self.text_shape = (text_len, text_dim)
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        z_ab = torch.randn(*self.shape, generator=g)
        f = self.shape[1]
        return {
            "z_cd": z_ab + 0.1 * torch.randn(*self.shape, generator=g),
            "z_ab": z_ab,
            "audio": torch.randn(*self.audio_shape, generator=g),
            "text": torch.randn(*self.text_shape, generator=g),
            "mouth": torch.tensor([0.5, 0.62]).expand(f, 2).clone(),
            "face_w": torch.full((f,), 0.45),
        }


class TimestepDependentSampler(IterableDataset):
    """Draws sigma first, then picks the data source from it (paper Eq. 3).

    sigma > threshold -> pseudo-paired data (stable pose/identity for structure formation);
    otherwise         -> arbitrary data (diverse lip motion and texture).
    With ``enabled=False`` (ablation) or no pseudo pairs, every sample comes from ``arbitrary``.
    """

    def __init__(self, pseudo_pairs: Dataset | None, arbitrary: Dataset, threshold: float = 0.85,
                 sigma_mode: str = "shifted_uniform", sigma_shift: float = 3.0, enabled: bool = True,
                 seed: int = 0) -> None:
        super().__init__()
        if len(arbitrary) == 0:
            raise ValueError("the arbitrary dataset is empty")
        self.pairs, self.arb = pseudo_pairs, arbitrary
        self.threshold, self.mode, self.shift = threshold, sigma_mode, sigma_shift
        self.enabled = enabled and pseudo_pairs is not None and len(pseudo_pairs) > 0
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rank = int(os.environ.get("RANK", 0))
        rng = random.Random(self.seed + 7919 * wid + 104729 * rank)
        g = torch.Generator().manual_seed(self.seed + 31 * wid + 1009 * rank)
        while True:
            sigma = float(sample_training_sigmas(1, mode=self.mode, shift=self.shift, generator=g)[0])
            use_pair = self.enabled and sigma > self.threshold
            ds = self.pairs if use_pair else self.arb
            sample = ds[rng.randrange(len(ds))]
            sample["sigma"] = torch.tensor(sigma)
            sample["is_pair"] = torch.tensor(use_pair)
            yield sample


def collate(batch: list[dict], text_max_len: int = TEXT_MAX_LEN) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "text"}
    out["text"] = pad_text([b["text"] for b in batch], text_max_len)
    return out
