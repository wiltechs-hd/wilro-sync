"""Pre-compute training clips (latents, Whisper windows, caption embeddings) into safetensors.

Manifest: a .jsonl with one job per line

    {"kind": "arbitrary",   "video": "hdtf/WDA_x_0.mp4", "caption": "...", "clips": 6}   # in-the-wild clip
    {"kind": "pseudo_pair", "cond": "mead/M003/video/front/neutral/level_1/001.mp4",
                            "target": "mead/M003/video/front/neutral/level_1/002.mp4"}  # MEAD pseudo pair

Optional per-row fields: ``id`` (stable clip name), ``clips`` (clips to cut from this video), ``av_offset``
(from ``wilro-sync sync-filter``; applied when cutting the audio windows), ``mouth`` [x, y] and ``face_w``.

* ``arbitrary``: (V_cd, V_ab) segment pairs are sampled from the same video; V_ab's audio is the target audio.
* ``pseudo_pair``: V_cd from ``cond`` and V_ab (+ its audio) from ``target``; same speaker and camera.

Incremental: clips are named after the source row, existing clip files are reused, and the index lists exactly
the clips of this manifest. Several manifests (e.g. stages) can therefore share one clip folder.
Frames are resampled to ``--fps`` and resized/centre-cropped to ``--size`` (use face-cropped videos).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ..io.media import read_audio, read_video
from ..models.audio import WhisperAudioEncoder, frame_audio_windows
from ..models.backbone import T5TextEncoder, WanVAE
from ..pipeline.lipsync import DEFAULT_PROMPT


def resize_crop(frames: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """uint8 [T, H, W, 3] -> float [3, T, h, w] in [-1, 1] (resize short side, centre crop)."""
    th, tw = size
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    h, w = x.shape[-2:]
    s = max(th / h, tw / w)
    nh, nw = max(th, round(h * s)), max(tw, round(w * s))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
    y0, x0 = (nh - th) // 2, (nw - tw) // 2
    return x[:, :, y0 : y0 + th, x0 : x0 + tw].permute(1, 0, 2, 3).contiguous()


class Encoders:
    """Frozen encoders. umT5 (~11 GB) is only resident while captions are encoded (``encode_captions``)."""

    def __init__(self, repo: str, audio_model: str, device: str) -> None:
        self.device = device
        self.vae = WanVAE.from_pretrained(repo, torch_dtype=torch.float32).to(device)
        self.whisper = WhisperAudioEncoder(audio_model, device=device)
        self._repo, self._cache = repo, {}

    def encode_captions(self, captions: list[str]) -> None:
        todo = [c for c in dict.fromkeys(captions) if c not in self._cache]
        if not todo:
            return
        t5 = T5TextEncoder.from_pretrained(self._repo, torch_dtype=torch.bfloat16, device=self.device)
        for c in todo:
            self._cache[c] = t5([c], trim=True)[0].to(torch.bfloat16).cpu()
        del t5
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def text(self, caption: str) -> torch.Tensor:
        if caption not in self._cache:
            self.encode_captions([caption])
        return self._cache[caption]

    def latent(self, clip: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(clip.unsqueeze(0).to(self.device))[0].to(torch.bfloat16).cpu()


def segment_starts(n_frames: int, length: int, k: int, rng: random.Random) -> list[tuple[int, int]]:
    """Up to k (a, c) pairs: V_ab = [a, a+L), V_cd = [c, c+L), non-overlapping when possible."""
    if n_frames < length:
        return []
    out = []
    for _ in range(k):
        a = rng.randrange(0, n_frames - length + 1)
        choices = [c for c in range(0, n_frames - length + 1) if abs(c - a) >= length]
        c = rng.choice(choices) if choices else rng.randrange(0, n_frames - length + 1)
        out.append((a, c))
    return out


def clip_base(row: dict) -> str:
    """Stable per-row name, so the same source video maps to the same clip files in every run/stage."""
    if row.get("id"):
        return str(row["id"]).replace("/", "_")
    key = "|".join(str(row.get(k, "")) for k in ("kind", "video", "cond", "target"))
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def main(argv: list[str] | None = None, encoders=None) -> int:
    p = argparse.ArgumentParser(prog="prepare_clips", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True, help="clip folder (shared between runs; existing clips are reused)")
    p.add_argument("--index", default=None, help="index to write (default: <out>/index.jsonl)")
    p.add_argument("--repo", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--audio-model", default="openai/whisper-tiny")
    p.add_argument("--audio-window", type=int, default=10)
    p.add_argument("--size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    p.add_argument("--frames", type=int, default=49, help="clip length, must be 1 + 4k")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--clips-per-video", type=int, default=4, help="default when a row has no 'clips' field")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if (a.frames - 1) % 4:
        raise SystemExit("--frames must be 1 + 4k")

    os.makedirs(a.out, exist_ok=True)
    index_path = a.index or os.path.join(a.out, "index.jsonl")
    index_dir = os.path.dirname(os.path.abspath(index_path))
    os.makedirs(index_dir, exist_ok=True)
    null_path = os.path.join(os.path.dirname(os.path.abspath(a.out)), "null_text.safetensors")
    rows = [json.loads(line) for line in open(a.manifest) if line.strip()]

    def names(row):
        n = 1 if row["kind"] == "pseudo_pair" else int(row.get("clips", a.clips_per_video))
        return [f"{row['kind']}_{clip_base(row)}_{j:02d}.safetensors" for j in range(n)]

    todo = [r for r in rows if not all(os.path.isfile(os.path.join(a.out, n)) for n in names(r))]
    enc = encoders
    if (todo or not os.path.isfile(null_path)) and enc is None:
        enc = Encoders(a.repo, a.audio_model, a.device)
    if enc is not None and hasattr(enc, "encode_captions"):  # encode all captions first, then free umT5
        need_null = not os.path.isfile(null_path)
        enc.encode_captions(([""] if need_null else []) + [r.get("caption", DEFAULT_PROMPT) for r in todo])
    if not os.path.isfile(null_path):
        save_file({"text": enc.text("")}, null_path)

    n_written = 0
    for r_i, row in enumerate(todo):
        try:
            rng = random.Random(f"{a.seed}:{clip_base(row)}")
            caption = row.get("caption", DEFAULT_PROMPT)
            mouth = torch.tensor(row.get("mouth", [0.5, 0.62]), dtype=torch.float32)
            n_clips = len(names(row))
            if row["kind"] == "arbitrary":
                path = os.path.join(a.root, row["video"])
                frames, fps = read_video(path, fps=a.fps)
                feats = enc.whisper(read_audio(path))
                jobs = [(frames, frames, ab, cd, feats)
                        for ab, cd in segment_starts(len(frames), a.frames, n_clips, rng)]
            elif row["kind"] == "pseudo_pair":
                cond, _ = read_video(os.path.join(a.root, row["cond"]), fps=a.fps)
                tpath = os.path.join(a.root, row["target"])
                tgt, fps = read_video(tpath, fps=a.fps)
                feats = enc.whisper(read_audio(tpath))
                if min(len(cond), len(tgt)) < a.frames:
                    continue
                ab = rng.randrange(0, len(tgt) - a.frames + 1)
                cd = rng.randrange(0, len(cond) - a.frames + 1)
                jobs = [(cond, tgt, ab, cd, feats)]
            else:
                raise ValueError(f"unknown kind {row['kind']!r}")
            for j, (src_cd, src_ab, ab, cd, feats) in enumerate(jobs):
                z_ab = enc.latent(resize_crop(src_ab[ab : ab + a.frames], tuple(a.size)))
                z_cd = enc.latent(resize_crop(src_cd[cd : cd + a.frames], tuple(a.size)))
                # SyncNet offset (from `wilro-sync sync-filter`): audio for frame v sits at v - av_offset
                shift = -float(row.get("av_offset", 0)) * a.fps / 25.0
                all_win = frame_audio_windows(feats, ab + a.frames, a.fps, window=a.audio_window, offset_frames=shift)
                f_lat = z_ab.shape[1]
                sample = {
                    "z_ab": z_ab, "z_cd": z_cd,
                    "audio": all_win[ab : ab + a.frames].to(torch.float16).contiguous(),
                    "text": enc.text(caption),
                    "mouth": mouth.expand(f_lat, 2).contiguous(),
                    "face_w": torch.full((f_lat,), float(row.get("face_w", 0.45))),
                }
                save_file(sample, os.path.join(a.out, f"{row['kind']}_{clip_base(row)}_{j:02d}.safetensors"))
                n_written += 1
        except Exception as e:  # keep going on bad files
            print(f"[skip] row {r_i}: {e}", file=sys.stderr)

    # index of every clip that exists for this manifest (new and reused)
    entries = []
    for row in rows:
        for n in names(row):
            fp = os.path.join(a.out, n)
            if os.path.isfile(fp):
                entries.append({"path": os.path.relpath(fp, index_dir), "kind": row["kind"]})
    tmp = index_path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in entries)
    os.replace(tmp, index_path)
    print(f"wrote {n_written} new clips ({len(entries)} in index) -> {index_path}")
    return len(entries)
