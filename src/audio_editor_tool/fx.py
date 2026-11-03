"""Transition effects layered over a join: riser, reverse swell, echo-out, filter sweep.

Every effect is timed to the song's beat and ends (or starts) exactly on the
downbeat of the incoming part.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

FX = ("riser", "swell", "echo", "sweep")
N_FFT = 1024


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def _band(x: np.ndarray, sr: int, lo: float | None, hi: float | None) -> np.ndarray:
    if lo:
        x = sosfiltfilt(butter(2, lo, "highpass", fs=sr, output="sos"), x, axis=-1)
    if hi:
        x = sosfiltfilt(butter(2, hi, "lowpass", fs=sr, output="sos"), x, axis=-1)
    return x


def _stft_mask(x: np.ndarray, sr: int, mask_fn) -> np.ndarray:
    """Apply a time-varying spectral gain mask_fn(freqs, t01) -> (freqs, frames)."""
    import librosa
    hop = N_FFT // 4
    out = []
    for ch in x:
        S = librosa.stft(ch, n_fft=N_FFT, hop_length=hop)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        t01 = np.linspace(0, 1, S.shape[1])
        out.append(librosa.istft(S * mask_fn(freqs, t01), hop_length=hop, length=len(ch)))
    return np.array(out, dtype=np.float32)


def riser(n: int, sr: int, level: float, rng: np.random.Generator) -> np.ndarray:
    """Filtered-noise sweep rising from ~200 Hz to ~8 kHz, getting louder, stops at the end."""
    noise = rng.standard_normal((2, n)).astype(np.float32)

    def mask(freqs, t):
        fc = 200 * 40 ** t                                   # sweep centre
        lf = np.log2(np.maximum(freqs, 1))[:, None]
        return np.exp(-0.5 * ((lf - np.log2(fc)[None, :]) / 0.9) ** 2)

    x = _stft_mask(noise, sr, mask)
    t = np.linspace(0, 1, n, dtype=np.float32)
    x *= t ** 2.5
    x[:, -int(0.01 * sr):] *= np.linspace(1, 0, int(0.01 * sr))
    return x * (level / _rms(x[:, -n // 8:]))


def _reverb_ir(sr: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal((2, n)) * np.exp(-6.9 * t / seconds)
    return _band(ir, sr, 200, 6000).astype(np.float32)


def swell(hit: np.ndarray, n: int, sr: int, level: float, rng: np.random.Generator) -> np.ndarray:
    """Reverse reverb of the incoming part's first beat, building into its downbeat."""
    hit = hit.copy()
    hit[:, -int(0.005 * sr):] *= np.linspace(1, 0, int(0.005 * sr))
    ir = _reverb_ir(sr, 2.5, rng)
    wet = np.array([fftconvolve(hit[c], ir[c]) for c in range(2)], dtype=np.float32)
    rev = wet[:, ::-1]
    x = rev[:, -n:] if rev.shape[1] >= n else np.pad(rev, ((0, 0), (n - rev.shape[1], 0)))
    x = _band(x, sr, 150, None) * np.linspace(0, 1, x.shape[1], dtype=np.float32) ** 1.5
    return x * (level / _rms(x[:, -n // 8:]))


def echo(tail: np.ndarray, sr: int, beat_s: float, level: float, repeats: int = 6) -> np.ndarray:
    """Tempo-synced echoes of the outgoing part's last beat, fading away."""
    src = _band(tail, sr, 300, 4500).astype(np.float32)
    f = int(0.005 * sr)
    src[:, :f] *= np.linspace(0, 1, f)
    src[:, -f:] *= np.linspace(1, 0, f)
    d = int(beat_s * sr)
    out = np.zeros((2, d * (repeats - 1) + src.shape[1]), np.float32)
    for k in range(repeats):
        g = 0.55 ** k
        rep = src if k % 2 == 0 else src[::-1]               # ping-pong left/right
        out[:, k * d:k * d + src.shape[1]] += g * rep
    return out * (level / _rms(src))


def sweep(x: np.ndarray, sr: int) -> np.ndarray:
    """High-pass sweep over the region: bass drains out, ending thin at the cut."""
    def mask(freqs, t):
        fc = 20 * 60 ** (t ** 2)                              # 20 Hz -> 1.2 kHz
        f = np.maximum(freqs, 1)[:, None]
        return 1 / np.sqrt(1 + (fc[None, :] / f) ** 8)
    return _stft_mask(x, sr, mask)


def apply(out: np.ndarray, audio: np.ndarray, sr: int, joins: list[tuple], beat_s: float,
          seed: int = 1) -> np.ndarray:
    """joins: (cut sample in output, end sample of outgoing part, start sample of incoming
    part in the original audio, Transition with .fx and .fx_level)."""
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    if out.shape[0] == 1:
        out = np.repeat(out, 2, axis=0)
    rng = np.random.default_rng(seed)
    beat = int(beat_s * sr)
    bar = 4 * beat
    for cut, a_end, b_start, tr in joins:
        effects, k = tr.fx, tr.fx_level
        if not effects:
            continue
        music = _rms(out[:, max(0, cut - 2 * bar):cut + 2 * bar])
        if "sweep" in effects:
            s0 = max(0, cut - bar)
            out[:, s0:cut] = sweep(out[:, s0:cut], sr)
        if "riser" in effects:
            n = min(2 * bar, cut)
            out[:, cut - n:cut] += riser(n, sr, 0.5 * k * music, rng)
        if "swell" in effects:
            n = min(2 * bar, cut)
            hit = audio[:, b_start:b_start + beat]
            if hit.shape[1] > 0:
                out[:, cut - n:cut] += swell(hit, n, sr, 0.6 * k * music, rng)
        if "echo" in effects:
            tail = audio[:, max(0, a_end - beat):a_end]
            if tail.shape[1] > 0:
                e = echo(tail, sr, beat_s, 0.28 * k * music)
                m = min(e.shape[1], out.shape[1] - cut)
                out[:, cut:cut + m] += e[:, :m]
    return out
