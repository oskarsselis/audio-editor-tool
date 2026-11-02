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


def overlap_len(a: Analysis, t_end: float, beats: float) -> float:
    """Overlap length ending at t_end that spans whole beats of the real beat grid."""
    near = a.beats[np.abs(a.beats - t_end) < 10]
    # Line fit over ~20 beats: precise even though each beat time is frame-quantized
    period = float(np.polyfit(np.arange(len(near)), near, 1)[0]) if len(near) > 4 else 60 / a.tempo
    start = _refine(a, t_end - beats * period, window=0.03)
    return t_end - start


def snap(a: Analysis, t: float, to: str = "bar") -> float:
    """Move a time to the nearest bar line or beat (then onto the drum hit)."""
    if to == "none":
        return t
    grid = np.r_[a.bar_starts, a.bar_ends[-1]] if to == "bar" else a.beats
    return _refine(a, float(grid[np.argmin(np.abs(grid - t))]))


def _blend(xa: np.ndarray, xb: np.ndarray, cut: int, sr: int, mode: str) -> np.ndarray:
    """Mix outgoing xa into incoming xb. `cut` = sample where xb's downbeat lands."""
    L = xa.shape[1]
    t = np.linspace(0, np.pi / 2, L, dtype=np.float32)
    fin, fout = np.sin(t), np.cos(t)
    if mode == "crossfade":
        return xa * fout + xb * fin
    # eq: bass swap. Mids/highs crossfade over the whole window; the bass of the
    # outgoing part stays until the downbeat and then hands over quickly, so two
    # bass lines never stack up.
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, 150, "lowpass", fs=sr, output="sos")
    la, lb = sosfiltfilt(sos, xa, axis=1), sosfiltfilt(sos, xb, axis=1)
    ha, hb = xa - la, xb - lb
    ramp = max(1, int(0.04 * sr))
    g = np.clip((np.arange(L) - (cut - ramp // 2)) / ramp, 0, 1).astype(np.float32)
    g_in, g_out = np.sin(g * np.pi / 2), np.cos(g * np.pi / 2)
    return ha * fout + hb * fin + la * g_out + lb * g_in


def _overlay(xa: np.ndarray, xb: np.ndarray, sr: int, hold: float) -> np.ndarray:
    """Old part's ending plays on top of the new part's start.

    The new part comes in at full volume right away; the old part keeps playing
    (full volume for `hold` of the window, then fades). The old part's bass drops
    out when the new part starts, so the low end stays clean.
    """
    from scipy.signal import butter, sosfiltfilt
    L = xa.shape[1]
    x = np.linspace(0, 1, L, dtype=np.float32)
    fade = np.where(x < hold, 1.0, np.cos(np.pi / 2 * (x - hold) / max(1 - hold, 1e-6)) ** 2)
    sos = butter(4, 150, "lowpass", fs=sr, output="sos")
    la = sosfiltfilt(sos, xa, axis=1)
    ramp = min(L, int(0.04 * sr))
    bass = np.zeros(L, np.float32)
    bass[:ramp] = np.cos(np.linspace(0, np.pi / 2, ramp)) ** 2
    b_in = np.ones(L, np.float32)
    b_in[:int(0.005 * sr)] = np.linspace(0, 1, int(0.005 * sr))
    return (xa - la) * fade + la * bass + xb * b_in


def _limit(x: np.ndarray, sr: int, ceiling: float = 0.98) -> np.ndarray:
    """Simple look-ahead peak limiter, so overlaps don't clip."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
    peak = np.abs(x).max(axis=0)
    if peak.max() <= ceiling:
        return x
    w = max(1, int(0.005 * sr))
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    # min over +-w then average over +-w: smooth, and never above what each sample needs
    gain = uniform_filter1d(minimum_filter1d(need, 2 * w + 1), 2 * w + 1)
    return (x * gain).astype(np.float32)


def render(audio: np.ndarray, sr: int, times: list[tuple[float, float]], natural_end: bool,
           xfade_s: float | list[float], fade_out_s: float, blend: str = "crossfade",
           position: str = "before", hold: float = 0.5) -> np.ndarray:
    """Join parts. position: where the blend sits relative to the cut —
    'before' (new part's downbeat hits at full volume), 'center', or 'after'
    (old part's tail, e.g. a held vocal, rings over the new part's start)."""
    n = len(times)
    Ls = [max(1, int(x * sr)) for x in (xfade_s if isinstance(xfade_s, list) else [xfade_s] * n)]
    if blend == "overlay":       # new part starts under the old part's last L samples
        position = "before"
    f = {"before": 1.0, "center": 0.5, "after": 0.0}[position]
    pieces = []
    for idx, (t0, t1) in enumerate(times):
        L = Ls[min(idx, len(Ls) - 1)]
        pre, post = int(round(f * L)), L - int(round(f * L))
        prev_L = Ls[idx - 1] if idx > 0 else 0
        prev_post = prev_L - int(round(f * prev_L))
        s, e = int(t0 * sr), int(t1 * sr)
        a = s + (prev_post if idx > 0 else 0) + (prev_L if idx > 0 and blend == "overlay" else 0)
        b = e - (pre if idx < n - 1 else 0)
        pieces.append(audio[:, a:b])
        if idx < n - 1:
            nb = int(times[idx + 1][0] * sr)
            xa = audio[:, e - pre:e + post]
            if blend == "overlay":
                xb = audio[:, nb:nb + L]
                m = min(xa.shape[1], xb.shape[1])
                pieces.append(_overlay(xa[:, :m], xb[:, :m], sr, hold))
                continue
            xb = audio[:, nb - pre:nb + post]
            m = min(xa.shape[1], xb.shape[1])
            pieces.append(_blend(xa[:, :m], xb[:, :m], pre, sr, blend))
    out = np.concatenate(pieces, axis=1).astype(np.float32)

    if times[0][0] > 0:                          # started mid-song: avoid a click
        r = min(int(0.01 * sr), out.shape[1])
        out[:, :r] *= np.linspace(0, 1, r, dtype=np.float32)
    if not natural_end:
        r = min(int(fade_out_s * sr), out.shape[1])
        out[:, -r:] *= np.cos(np.linspace(0, np.pi / 2, r, dtype=np.float32)) ** 2
    else:
        r = min(int(0.01 * sr), out.shape[1])
        out[:, -r:] *= np.linspace(1, 0, r, dtype=np.float32)
    return _limit(out, sr)
