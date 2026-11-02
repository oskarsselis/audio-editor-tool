"""Stitches the chosen segments together with beat-aligned crossfades."""

from __future__ import annotations

import numpy as np

from .analysis import Analysis
from .planner import Plan


def _refine(a: Analysis, t: float, window: float = 0.04) -> float:
    """Snap a cut time to the nearest strong onset (the actual drum hit) if one is close."""
    env, hop = a.onset_env, a.onset_hop_s
    c = int(round(t / hop))
    w = int(window / hop)
    lo, hi = max(0, c - w), min(len(env), c + w + 1)
    if hi <= lo:
        return t
    seg = env[lo:hi]
    if seg.max() < 2 * np.median(env):
        return t
    return (lo + int(np.argmax(seg))) * hop - 0.005   # land just before the transient


def segment_times(a: Analysis, p: Plan) -> list[tuple[float, float]]:
    times = []
    for idx, (s, e) in enumerate(p.segments):
        t0 = 0.0 if s == 0 else _refine(a, a.bar_starts[s])
        last = idx == len(p.segments) - 1
        t1 = a.duration if (last and p.natural_end) else _refine(a, a.bar_ends[e])
        times.append((t0, t1))
    return times


def render(audio: np.ndarray, sr: int, a: Analysis, p: Plan,
           xfade_s: float, fade_out_s: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
    times = segment_times(a, p)
    xf = max(1, int(xfade_s * sr))
    t = np.linspace(0, np.pi / 2, xf, dtype=np.float32)
    fade_in, fade_out = np.sin(t), np.cos(t)

    s0, e0 = (int(x * sr) for x in times[0])
    out = audio[:, s0:e0].copy()
    if s0 > 0:                                   # started mid-song: avoid a click
        ramp = min(int(0.01 * sr), out.shape[1])
        out[:, :ramp] *= np.linspace(0, 1, ramp, dtype=np.float32)

    for t0, t1 in times[1:]:
        s, e = int(t0 * sr), int(t1 * sr)
        L = min(xf, s, out.shape[1])
        # Crossfade over the moments *before* the downbeat, so the new section's
        # downbeat hits at full volume.
        out[:, -L:] = out[:, -L:] * fade_out[-L:] + audio[:, s - L:s] * fade_in[-L:]
        out = np.concatenate([out, audio[:, s:e]], axis=1)

    if not p.natural_end:
        L = min(int(fade_out_s * sr), out.shape[1])
        curve = np.cos(np.linspace(0, np.pi / 2, L, dtype=np.float32)) ** 2
        out[:, -L:] *= curve
    else:
        L = min(int(0.01 * sr), out.shape[1])
        out[:, -L:] *= np.linspace(1, 0, L, dtype=np.float32)
    return out, times
