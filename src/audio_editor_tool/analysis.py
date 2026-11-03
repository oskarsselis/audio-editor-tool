"""Beat/bar detection and per-bar features used to find good cut points."""

from __future__ import annotations

import io
from dataclasses import dataclass, fields

import librosa
import numpy as np

SR = 22050
HOP = 512


@dataclass
class Analysis:
    duration: float
    tempo: float
    beats: np.ndarray          # beat times (s)
    bar_starts: np.ndarray     # start time of each bar (s), length n
    bar_ends: np.ndarray       # end time of each bar (s), length n
    sim: np.ndarray            # (n, n) bar-to-bar similarity, 0..1
    texture: np.ndarray        # (n, n) similarity of the surrounding few bars (for sections)
    score: np.ndarray          # (n,) how "important" a bar is (energy + repetition), 0..1
    onset_env: np.ndarray      # fine onset envelope for cut refinement
    onset_hop_s: float

    @property
    def n(self) -> int:
        return len(self.bar_starts)

    @property
    def bar_len(self) -> float:
        return float(np.median(self.bar_ends - self.bar_starts))

    def bar_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.bar_starts, t, side="right") - 1, 0, self.n - 1))

    def beat_period(self, t: float) -> float:
        """Beat length (s) around time t. A line fit over ~20 beats is precise even
        though each detected beat time is rounded to an analysis frame."""
        near = self.beats[np.abs(self.beats - t) < 10]
        if len(near) > 4:
            return float(np.polyfit(np.arange(len(near)), near, 1)[0])
        return 60 / self.tempo

    def to_npz(self) -> bytes:
        """Serialize (numbers only, no pickle) so a project opens without re-analysis."""
        buf = io.BytesIO()
        np.savez_compressed(buf, **{f.name: np.asarray(getattr(self, f.name)) for f in fields(self)})
        return buf.getvalue()

    @classmethod
    def from_npz(cls, data: bytes) -> "Analysis":
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            vals = {f.name: z[f.name] for f in fields(cls)}
        for k in ("duration", "tempo", "onset_hop_s"):
            vals[k] = float(vals[k])
        return cls(**vals)


def _cosine(x: np.ndarray) -> np.ndarray:
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return x @ x.T


def _rank01(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def analyze(audio: np.ndarray, sr: int, beats_per_bar: int = 4) -> Analysis:
    y = librosa.resample(librosa.to_mono(audio), orig_sr=sr, target_sr=SR)
    duration = len(y) / SR

    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beat_frames) < beats_per_bar * 4:
        raise RuntimeError("Could not detect a steady beat in this audio.")
    beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)

    # Features
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=20)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=64)
    low = librosa.power_to_db(mel[:6].sum(axis=0))          # kick/bass energy
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]

    # Downbeat phase: beats with strongest bass hits + chord changes
    chroma_change = np.r_[0, np.linalg.norm(np.diff(chroma, axis=1), axis=0)]
    def z(v): return (v - v.mean()) / (v.std() + 1e-9)
    beat_strength = z(low[beat_frames]) + z(chroma_change[beat_frames]) + 0.5 * z(onset[beat_frames])
    phase = int(np.argmax([beat_strength[p::beats_per_bar].mean() for p in range(beats_per_bar)]))
    down = beat_frames[phase::beats_per_bar]

    bar_len_frames = int(np.median(np.diff(down)))
    total_frames = chroma.shape[1]
    # Quiet intros/outros often have no detected beats: extend the bar grid over them
    tol = bar_len_frames // 8
    head = list(range(down[0] - bar_len_frames, -tol - 1, -bar_len_frames))[::-1]
    tail = list(range(down[-1] + bar_len_frames, total_frames - bar_len_frames // 2, bar_len_frames))
    down = np.r_[np.maximum(head, 0), down, tail].astype(int)
    starts = down
    ends = np.r_[down[1:], min(down[-1] + bar_len_frames, total_frames)]
    keep = ends - starts > bar_len_frames // 2
    starts, ends = starts[keep], ends[keep]

    # Per-bar features: chroma split into beat-sized slices (captures the progression),
    # mean timbre, loudness.
    ch_feats, mf_feats, loud = [], [], []
    for s, e in zip(starts, ends):
        edges = np.linspace(s, e, beats_per_bar + 1).astype(int)
        ch_feats.append(np.concatenate(
            [chroma[:, a:max(b, a + 1)].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])]))
        mf_feats.append(mfcc[1:, s:e].mean(axis=1))
        loud.append(rms[s:e].mean())
    ch_feats, mf_feats = np.array(ch_feats), np.array(mf_feats)
    loud_db = librosa.amplitude_to_db(np.array(loud) + 1e-9)

    mf_std = (mf_feats - mf_feats.mean(0)) / (mf_feats.std(0) + 1e-9)
    s_ch = _cosine(ch_feats)
    s_mf = (_cosine(mf_std) + 1) / 2
    s_ld = 1 - np.minimum(np.abs(loud_db[:, None] - loud_db[None, :]) / 12, 1)
    sim = 0.5 * s_ch + 0.3 * s_mf + 0.2 * s_ld

    # Texture: harmony averaged over the 4-bar phrase + timbre + loudness
    n = len(starts)
    bar_chroma = ch_feats.reshape(n, beats_per_bar, 12).mean(axis=1)
    phrase_chroma = np.array([bar_chroma[max(0, i - 1):i + 3].mean(axis=0) for i in range(n)])
    texture = 0.3 * _cosine(phrase_chroma) + 0.4 * s_mf + 0.3 * s_ld

    # Importance: loud bars and bars that repeat elsewhere (choruses/hooks) score high
    far = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) >= 8
    rep = np.array([np.sort(sim[i][far[i]])[-3:].mean() if far[i].any() else 0 for i in range(n)])
    score = 0.6 * _rank01(loud_db) + 0.4 * _rank01(rep)

    fine_hop = 128
    onset_fine = librosa.onset.onset_strength(y=y, sr=SR, hop_length=fine_hop)

    return Analysis(
        duration=duration,
        tempo=tempo,
        beats=beats,
        bar_starts=librosa.frames_to_time(starts, sr=SR, hop_length=HOP),
        bar_ends=np.minimum(librosa.frames_to_time(ends, sr=SR, hop_length=HOP), duration),
        sim=sim,
        texture=texture,
        score=score,
        onset_env=onset_fine,
        onset_hop_s=fine_hop / SR,
    )


def sections(a: Analysis, min_bars: int = 4) -> list[tuple[int, int, str]]:
    """Rough song map: list of (first_bar, last_bar, label). Same label = similar music."""
    n = a.n
    # Novelty along the diagonal (checkerboard kernel) to find section boundaries
    s = (a.texture - a.texture.mean()) / (a.texture.std() + 1e-9)
    nov = np.zeros(n)
    for i in range(2, n - 1):
        k = min(4, i, n - i)
        kernel = np.kron(np.array([[1, -1], [-1, 1]]), np.ones((k, k)))
        nov[i] = (s[i - k:i + k, i - k:i + k] * kernel).sum() / (4 * k * k)
    bounds = [0]
    thresh = 0.25 * nov.max()
    for i in np.argsort(-nov):
        local_peak = nov[i] >= nov[max(0, i - 2):i + 3].max()
        if (local_peak and nov[i] > thresh and min_bars <= i <= n - 2
                and all(abs(i - b) >= min_bars for b in bounds)):
            bounds.append(int(i))
    bounds = sorted(bounds) + [n]
    segs = list(zip(bounds[:-1], [b - 1 for b in bounds[1:]]))

    labels: list[str] = []
    for idx, (s, e) in enumerate(segs):
        label = None
        for j in range(idx):
            ps, pe = segs[j]
            m = min(e - s, pe - ps) + 1
            diag = np.mean([a.sim[s + t, ps + t] for t in range(m)])
            if diag > 0.9:
                label = labels[j]
                break
        labels.append(label or chr(ord("A") + len(set(labels))))
    return [(s, e, lab) for (s, e), lab in zip(segs, labels)]
