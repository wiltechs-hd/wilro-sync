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
    """Per-window weights over its real frames; weights of overlapping windows sum to 1.

    Where a window starts inside the previous one, the two cross-fade over exactly ``overlap`` frames: the last
    ``overlap`` frames they share (linear ramp). Shared frames before the fade come from the earlier window only.
    This matters for the last window of a segment, which is shifted back to end on the last frame and can share
    far more than ``overlap`` frames; those frames are never averaged 50/50 (two different generated mouths at
    half strength look like a doubled mouth). ``overlap=0`` is a hard cut."""
    raw = [torch.ones(w.real_len) for w in windows]
    order = sorted(range(len(windows)), key=lambda i: (windows[i].start, windows[i].end))
    for ia, ib in zip(order, order[1:]):
        a, b = windows[ia], windows[ib]
        if b.start >= a.end:  # no shared frames (next segment, or windows that just touch)
            continue
        shared_end = min(a.end, b.end)
        k = min(overlap, shared_end - b.start)
        fade_start = shared_end - k
        raw[ib][: fade_start - b.start] = 0.0  # shared, before the fade: the earlier window only
        if k > 0:
            ramp = torch.linspace(0, 1, k + 2)[1:-1]
            raw[ib][fade_start - b.start: shared_end - b.start] *= ramp
            raw[ia][fade_start - a.start: shared_end - a.start] *= 1 - ramp
    total = torch.zeros(num_frames)
    for w, wt in zip(windows, raw):
        total[w.start : w.end] += wt
    return [wt / total[w.start : w.end].clamp_min(1e-8) for w, wt in zip(windows, raw)]
