"""Generate a synthetic test song with a known structure.

Usage: uv run python examples/make_demo_song.py demo_song.wav
Structure (122 BPM, 4/4): intro 4 | verse 8 | chorus 8 | verse 8 | chorus 8 | bridge 8 | chorus 8 | outro 4
"""

import sys

import numpy as np
import soundfile as sf

SR = 44100
BPM = 122
BEAT = 60 / BPM
BAR = 4 * BEAT
rng = np.random.default_rng(0)

NOTES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def freq(name: str, octave: int) -> float:
    return 440 * 2 ** ((NOTES[name] + 12 * (octave + 1) - 69) / 12)


def chord(root: str, minor: bool) -> list[float]:
    r = freq(root, 3)
    third = 3 if minor else 4
    return [r, r * 2 ** (third / 12), r * 2 ** (7 / 12)]


def env(n: int, attack: float = 0.01, decay: float = 4.0) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(t / attack, 1) * np.exp(-decay * t)


def tone(f: float, dur: float, kind: str = "saw", decay: float = 2.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    if kind == "sine":
        w = np.sin(2 * np.pi * f * t)
    else:
        w = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 8))
    return w * env(n, 0.02, decay)


def kick() -> np.ndarray:
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * (50 + 100 * np.exp(-30 * t)) * t) * np.exp(-12 * t)


def snare() -> np.ndarray:
    n = int(0.2 * SR)
    return rng.standard_normal(n) * np.exp(-25 * np.arange(n) / SR) * 0.5


def hat() -> np.ndarray:
    n = int(0.05 * SR)
    x = np.diff(rng.standard_normal(n + 1))
    return x * np.exp(-80 * np.arange(n) / SR) * 0.15


def add(buf: np.ndarray, x: np.ndarray, t: float, gain: float = 1.0) -> None:
    i = int(t * SR)
    j = min(len(buf), i + len(x))
    buf[i:j] += gain * x[: j - i]


SECTIONS = [
    ("intro", 4, [("A", 1), ("F", 0), ("C", 0), ("G", 0)], 0),
    ("verse", 8, [("A", 1), ("F", 0), ("C", 0), ("G", 0)], 1),
    ("chorus", 8, [("C", 0), ("G", 0), ("A", 1), ("F", 0)], 2),
    ("verse", 8, [("A", 1), ("F", 0), ("C", 0), ("G", 0)], 1),
    ("chorus", 8, [("C", 0), ("G", 0), ("A", 1), ("F", 0)], 2),
    ("bridge", 8, [("D", 1), ("E", 1), ("F", 0), ("G", 0)], 1),
    ("chorus", 8, [("C", 0), ("G", 0), ("A", 1), ("F", 0)], 2),
    ("outro", 4, [("A", 1), ("F", 0), ("C", 0), ("C", 0)], 0),
]
MELODY = [("E", 5), ("G", 5), ("A", 5), ("G", 5), ("D", 5), ("E", 5), ("C", 5), ("D", 5)]


def main(out: str) -> None:
    total_bars = sum(s[1] for s in SECTIONS)
    buf = np.zeros(int((total_bars * BAR + 3) * SR))
    bar = 0
    for name, n_bars, prog, level in SECTIONS:
        for b in range(n_bars):
            t0 = (bar + b) * BAR
            root, minor = prog[b % len(prog)]
            last_outro = name == "outro" and b == n_bars - 1
            for f in chord(root, bool(minor)):
                add(buf, tone(f, BAR * (2.5 if last_outro else 1.05), decay=1.0 if last_outro else 1.5),
                    t0, 0.08)
            for beat in range(4):
                tb = t0 + beat * BEAT
                add(buf, tone(freq(root, 2), BEAT * 0.9, "sine", 4), tb, 0.25 if level else 0.12)
                if level >= 1 and not last_outro:
                    if beat in (0, 2) or level == 2:
                        add(buf, kick(), tb, 0.8)
                    if beat in (1, 3):
                        add(buf, snare(), tb, 0.6 if level == 2 else 0.4)
                    for h in range(2 if level == 1 else 4):
                        add(buf, hat(), tb + h * BEAT / (2 if level == 1 else 4))
            if name == "chorus":
                for s in range(2):
                    nm, octv = MELODY[(b * 2 + s) % len(MELODY)]
                    add(buf, tone(freq(nm, octv), BEAT * 2, "saw", 3), t0 + s * 2 * BEAT, 0.07)
        bar += n_bars
    buf /= np.abs(buf).max() * 1.1
    stereo = np.stack([buf, np.roll(buf, 30)], axis=1)
    sf.write(out, stereo, SR)
    t = 0.0
    for name, n_bars, *_ in SECTIONS:
        print(f"{int(t // 60)}:{t % 60:04.1f}  {name}")
        t += n_bars * BAR
    print(f"{int(t // 60)}:{t % 60:04.1f}  end")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_song.wav")
