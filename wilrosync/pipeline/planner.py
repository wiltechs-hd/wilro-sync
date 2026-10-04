"""Clip planner: splits a long video into overlapping generation windows.

* Windows only cover frames where the target is present; other frames pass through unchanged.
* Every window has a Wan-compatible length ``1 + 4k``. Segments shorter than that are padded at
  the end (repeat last frame) and trimmed after generation.
* Overlapping windows are blended with linear cross-fade weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def valid_length(n: int, temporal_factor: int = 4) -> int:
    """Smallest length >= n of the form 1 + temporal_factor * k."""
    if n <= 1:
        return 1
    k = -(-(n - 1) // temporal_factor)
    return 1 + temporal_factor * k


@dataclass
class Window:
    start: int  # first source frame
    end: int  # one past the last real source frame
    length: int  # generated length (>= end - start, 1 + 4k)

    @property
    def real_len(self) -> int:
        return self.end - self.start


def presence_segments(presence: torch.Tensor, min_len: int = 1) -> list[tuple[int, int]]:
    segs, start = [], None
    p = presence.tolist()
    for i, v in enumerate(p + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    return segs


def plan_windows(
    num_frames: int,
    window: int = 81,
    overlap: int = 12,
    presence: torch.Tensor | None = None,
    temporal_factor: int = 4,
    min_segment: int = 2,
) -> list[Window]:
    if (window - 1) % temporal_factor != 0:
        raise ValueError(f"window must be 1 + {temporal_factor}k, got {window}")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be in [0, window)")
    if presence is None:
        segments = [(0, num_frames)]
    else:
        segments = presence_segments(presence[:num_frames], min_len=min_segment)
    out: list[Window] = []
    stride = window - overlap
    for s, e in segments:
        n = e - s
        if n <= window:
            out.append(Window(s, e, valid_length(n, temporal_factor)))
            continue
        starts = list(range(s, e - window + 1, stride))
        if starts[-1] + window < e:
            starts.append(e - window)
        out.extend(Window(st, st + window, window) for st in starts)
    return out


def blend_weights(windows: list[Window], num_frames: int, overlap: int) -> list[torch.Tensor]:
    """Per-window weights over its real frames; weights of overlapping windows sum to 1."""
    raw = []
    for w in windows:
        wt = torch.ones(w.real_len)
        if overlap > 0:
            ramp = torch.linspace(0, 1, overlap + 2)[1:-1]
            has_prev = any(o is not w and o.start < w.start < o.end for o in windows)
            has_next = any(o is not w and o.start < w.end <= o.end and o.start > w.start for o in windows)
            if has_prev:
                k = min(overlap, w.real_len)
                wt[:k] = torch.minimum(wt[:k], ramp[:k])
            if has_next:
                k = min(overlap, w.real_len)
                wt[-k:] = torch.minimum(wt[-k:], ramp.flip(0)[-k:])
        raw.append(wt)
    total = torch.zeros(num_frames)
    for w, wt in zip(windows, raw):
        total[w.start : w.end] += wt
    return [wt / total[w.start : w.end].clamp_min(1e-8) for w, wt in zip(windows, raw)]
