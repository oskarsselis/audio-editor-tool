"""Chooses which bars to keep so the edit hits the target length with smooth jumps.

Dynamic programming over bars: a path either plays the next bar or jumps forward,
landing on a bar that sounds like the one it replaces. Every jump lands on a block
of at least `min_segment` bars so the result never sounds choppy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import Analysis

NEG = -1e18


@dataclass
class Settings:
    target: float                      # seconds
    keep: list[tuple[float, float]] = field(default_factory=list)
    skip: list[tuple[float, float]] = field(default_factory=list)
    start: str = "intro"               # intro | any
    end: str = "auto"                  # auto | outro | fade
    min_segment: int = 4               # bars
    tolerance: float = 4.0             # seconds; beyond this length errors cost a lot
    xfade_s: float = 0.25


@dataclass
class Plan:
    segments: list[tuple[int, int]]    # (first_bar, last_bar), inclusive
    natural_end: bool
    est_duration: float
    join_match: list[float]            # 0..1 per transition


def _shift(m: np.ndarray, di: int, dj: int) -> np.ndarray:
    """out[i, j] = m[i + di, j + dj] (nan when out of range)."""
    n = m.shape[0]
    out = np.full_like(m, np.nan)
    i0, i1 = max(0, -di), min(n, n - di)
    j0, j1 = max(0, -dj), min(n, n - dj)
    out[i0:i1, j0:j1] = m[i0 + di:i1 + di, j0 + dj:j1 + dj]
    return out


def jump_match(a: Analysis) -> np.ndarray:
    """match[i, j]: how well 'play bar i, then jump to bar j' imitates the original.

    Compares bar j with bar i+1 (what would have played) and bar j-1 with bar i
    (lead-in), plus one more bar of context each side.
    """
    s = a.sim
    stack = np.stack([_shift(s, 1, 0), _shift(s, 2, 1), _shift(s, 0, -1), _shift(s, -1, -2)])
    w = np.array([0.35, 0.15, 0.35, 0.15])[:, None, None]
    valid = ~np.isnan(stack)
    return np.nansum(np.nan_to_num(stack) * w, axis=0) / np.maximum((valid * w).sum(axis=0), 1e-9)


def _mask(a: Analysis, ranges: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(a.n, bool)
    for t0, t1 in ranges:
        overlap = np.minimum(a.bar_ends, t1) - np.maximum(a.bar_starts, t0)
        m |= overlap >= 0.5 * (a.bar_ends - a.bar_starts)
        m[a.bar_at((t0 + t1) / 2)] = True
    return m


def plan(a: Analysis, cfg: Settings, section_starts: set[int] | None = None) -> Plan:
    n, m = a.n, max(1, cfg.min_segment)
    section_starts = set(section_starts or {0}) | {n}
    d = a.bar_ends - a.bar_starts
    preroll = a.bar_starts[0]
    tail = a.duration - a.bar_ends[-1]
    keep, skip = _mask(a, cfg.keep), _mask(a, cfg.skip)
    keep &= ~skip
    keepcum = np.r_[0, np.cumsum(keep)]
    w_score = 0.3

    # Jump cost: poor match, broken phrase grid (skips that aren't multiples of 4/8 bars),
    # plus a flat cost per jump so fewer, cleaner cuts win.
    match = jump_match(a)
    dis = 1 - match
    iu = np.triu_indices(n, 2)
    lo, mid = np.percentile(dis[iu], 5), np.percentile(dis[iu], 50)
    jnorm = np.clip((dis - lo) / max(mid - lo, 1e-6), 0, 3)
    gap = np.arange(n)[None, :] - np.arange(n)[:, None] - 1
    phrase = np.select([gap % 8 == 0, gap % 4 == 0, gap % 2 == 0], [0.0, 0.2, 0.6], 1.0)
    # Cutting out whole sections (cut at a section end, land on a section start) sounds natural
    is_bound = np.zeros(n + 1, bool)
    is_bound[[b for b in section_starts if b <= n]] = True
    at_end, at_start = is_bound[1:n + 1][:, None], is_bound[:n][None, :]
    section_bonus = 0.4 * (at_end & at_start) + 0.1 * (at_end ^ at_start)
    cost = jnorm + phrase + 0.4 - section_bonus

    # Landing blocks: bars j..j+m-1 played together
    bscore = np.array([a.score[j:j + m].sum() if j + m <= n else 0 for j in range(n)])
    bdur = np.array([d[j:j + m].sum() if j + m <= n else 0 for j in range(n)])
    bok = np.array([j + m <= n and not skip[j:j + m].any() for j in range(n)])

    V = np.full((n + 1, n), NEG)
    D = np.zeros((n + 1, n))
    # back[(k, i)] = (prev_k, prev_i, bars added); seeds have prev = None
    back: dict[tuple[int, int], tuple[int | None, int | None, list[int]]] = {}

    first_keep = int(np.argmax(keep)) if keep.any() else n - 1
    if cfg.start == "intro" and bok[0]:
        starts = [0]
    else:
        starts = [j for j in range(min(first_keep, n - m) + 1) if bok[j]]
    for j in starts:
        pen = 0 if j == 0 else 0.3 + 0.5 * (j not in section_starts)
        k, last = m, j + m - 1
        V[k, last] = w_score * bscore[j] - pen
        D[k, last] = bdur[j] + (preroll if j == 0 else 0)
        back[(k, last)] = (None, None, list(range(j, last + 1)))

    for k in range(1, n + 1):
        for i in np.nonzero(V[k] > NEG / 2)[0]:
            v, dur = V[k, i], D[k, i]
            # Play next bar
            j = i + 1
            if j < n and k + 1 <= n and not skip[j]:
                cand = v + w_score * a.score[j]
                if cand > V[k + 1, j]:
                    V[k + 1, j], D[k + 1, j] = cand, dur + d[j]
                    back[(k + 1, j)] = (k, i, [j])
            # Jump forward to bar j (no kept bars may be skipped)
            if k + m > n:
                continue
            js = np.arange(i + 2, n - m + 1)
            if not len(js):
                continue
            ok = bok[js] & (keepcum[js] - keepcum[i + 1] == 0)
            js = js[ok]
            if not len(js):
                continue
            cand = v - cost[i, js] + w_score * bscore[js]
            tgt = js + m - 1
            better = cand > V[k + m, tgt]
            for jj, t, c in zip(js[better], tgt[better], cand[better]):
                V[k + m, t] = c
                D[k + m, t] = dur + bdur[jj] - cfg.xfade_s
                back[(k + m, t)] = (k, i, list(range(jj, t + 1)))

    def length_penalty(x: float) -> float:
        err = abs(x - cfg.target)
        return 0.15 * err + 1.0 * max(0.0, err - cfg.tolerance)

    best, best_val = None, NEG
    for k in range(1, n + 1):
        for i in np.nonzero(V[k] > NEG / 2)[0]:
            if keepcum[n] - keepcum[i + 1] > 0:      # kept bars still ahead
                continue
            options = []
            if i == n - 1 and cfg.end in ("auto", "outro"):
                options.append((True, D[k, i] + tail, 0.0))
            if cfg.end == "fade" or (cfg.end == "auto" and i < n - 1):
                pen = 0.6 + 0.4 * ((i + 1) not in section_starts)
                options.append((False, D[k, i], pen))
            for natural, dur, pen in options:
                val = V[k, i] - pen - length_penalty(dur)
                if val > best_val:
                    best_val, best = val, (k, i, natural, dur)
    if best is None:
        raise RuntimeError("No valid edit found - try a longer target or fewer --keep ranges.")

    k, i, natural, dur = best
    bars: list[int] = []
    node: tuple | None = (k, i)
    while node is not None:
        pk, pi, added = back[node]
        bars = added + bars
        node = (pk, pi) if pk is not None else None
    segs: list[tuple[int, int]] = []
    for b in bars:
        if segs and b == segs[-1][1] + 1:
            segs[-1] = (segs[-1][0], b)
        else:
            segs.append((b, b))
    joins = [float(match[segs[t][1], segs[t + 1][0]]) for t in range(len(segs) - 1)]
    return Plan(segments=segs, natural_end=natural, est_duration=float(dur), join_match=joins)
