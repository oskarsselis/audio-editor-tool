"""Stitches the chosen segments together with beat-aligned crossfades."""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class Transition:
    """How one part flows into the next."""
    blend: str = "crossfade"            # crossfade | eq | overlay
    xfade_s: float = 0.25               # blend / overlap length in seconds
    position: str = "before"            # before | center | after (not used by overlay)
    hold: float = 0.5                   # overlay: share of overlap at full volume
    fx: list[str] = field(default_factory=list)   # riser, swell, echo, sweep
    fx_level: float = 1.0               # effect volume multiplier


def render(audio: np.ndarray, sr: int, times: list[tuple[float, float]], natural_end: bool,
           xfade_s: float | list[float], fade_out_s: float, blend: str = "crossfade",
           position: str = "before", hold: float = 0.5,
           fx: list[str] | None = None, beat_s: float | None = None, fade_in_s: float = 0.0,
           gains_db: list[float] | None = None, master_db: float = 0.0) -> np.ndarray:
    """Same settings for every join (CLI). See render_parts for per-join settings."""
    n = len(times)
    xs = xfade_s if isinstance(xfade_s, list) else [xfade_s] * n
    trans = [Transition(blend, xs[i], position, hold, list(fx or [])) for i in range(n - 1)]
    out, _ = render_parts(audio, sr, times, trans, natural_end, fade_out_s, beat_s,
                          fade_in_s=fade_in_s, gains_db=gains_db, master_db=master_db)
    return out


def render_parts(audio: np.ndarray, sr: int, times: list[tuple[float, float]],
                 transitions: list[Transition], natural_end: bool, fade_out_s: float,
                 beat_s: float | None = None, fade_in_s: float = 0.0,
                 gains_db: list[float] | None = None, master_db: float = 0.0
                 ) -> tuple[np.ndarray, list[float]]:
    """Join parts with per-join settings. Returns (audio, cut times in the output in s).

    gains_db: volume change per part in dB (0 = unchanged); master_db: whole output.
    Gain changes between parts happen inside the blend, so they are smooth.
    """
    n = len(times)
    g = [10 ** (db / 20) for db in (gains_db or [])] + [1.0] * n
    pieces: list[np.ndarray] = []
    joins = []                     # (cut in output, outgoing end, incoming start) in samples
    length = 0
    skip_in = 0                    # samples of the next part already used by the blend
    for idx, (t0, t1) in enumerate(times):
        s, e = int(t0 * sr), int(t1 * sr)
        if idx == n - 1:
            pieces.append(audio[:, s + skip_in:e] * g[idx])
            length += pieces[-1].shape[1]
            break
        tr = transitions[idx]
        L = max(1, int(tr.xfade_s * sr))
        nb = int(times[idx + 1][0] * sr)
        if tr.blend == "overlay":
            pre, post = L, 0
        else:
            f = {"before": 1.0, "center": 0.5, "after": 0.0}[tr.position]
            pre, post = int(round(f * L)), L - int(round(f * L))
        body = audio[:, s + skip_in:max(s + skip_in, e - pre)] * g[idx]
        pieces.append(body)
        length += body.shape[1]
        xa = audio[:, e - pre:e + post] * g[idx]
        if tr.blend == "overlay":
            xb = audio[:, nb:nb + L] * g[idx + 1]
            m = min(xa.shape[1], xb.shape[1])
            mix = _overlay(xa[:, :m], xb[:, :m], sr, tr.hold)
            joins.append((length, e, nb, tr))
            skip_in = m
        else:
            xb = audio[:, max(0, nb - pre):nb + post] * g[idx + 1]
            m = min(xa.shape[1], xb.shape[1])
            mix = _blend(xa[:, :m], xb[:, :m], min(pre, m), sr, tr.blend)
            joins.append((length + pre, e, nb, tr))
            skip_in = post
        pieces.append(mix)
        length += mix.shape[1]
    out = np.concatenate(pieces, axis=1).astype(np.float32)
    if master_db:
        out *= np.float32(10 ** (master_db / 20))
    if any(j[3].fx for j in joins):
        from .fx import apply
        out = apply(out, audio, sr, joins, beat_s or 0.5)

    if fade_in_s > 0:
        r = min(int(fade_in_s * sr), out.shape[1])
        out[:, :r] *= np.sin(np.linspace(0, np.pi / 2, r, dtype=np.float32)) ** 2
    elif times[0][0] > 0:                        # started mid-song: avoid a click
        r = min(int(0.01 * sr), out.shape[1])
        out[:, :r] *= np.linspace(0, 1, r, dtype=np.float32)
    if not natural_end:
        r = min(int(fade_out_s * sr), out.shape[1])
        out[:, -r:] *= np.cos(np.linspace(0, np.pi / 2, r, dtype=np.float32)) ** 2
    else:
        r = min(int(0.01 * sr), out.shape[1])
        out[:, -r:] *= np.linspace(1, 0, r, dtype=np.float32)
    return _limit(out, sr), [j[0] / sr for j in joins]
