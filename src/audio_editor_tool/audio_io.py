"""Loading and saving audio in most formats (ffmpeg used as fallback)."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (audio[channels, samples] float32, sample_rate)."""
    path = str(path)
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"Cannot read {path}: install ffmpeg for this format")
        wav = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "wav", "-acodec", "pcm_f32le", "-"],
            check=True, capture_output=True,
        ).stdout
        data, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), sr


def save(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Save audio[channels, samples]. wav/flac/ogg natively, others via ffmpeg."""
    path = Path(path)
    audio = np.clip(audio, -1.0, 1.0).T
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        sf.write(path, audio, sr)
        return
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"Saving {path.suffix} needs ffmpeg (or use .wav/.flac)")
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, sr, subtype="FLOAT")
        extra = ["-b:a", "320k"] if path.suffix.lower() == ".mp3" else []
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", tmp.name, *extra, str(path)], check=True
        )
