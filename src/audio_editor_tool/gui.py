"""Visual editor: a local web app that opens in its own window.

Run `audio-editor-tool-gui`. Everything stays on your computer; the server only
listens on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

import numpy as np

from . import audio_io

WORKDIR = Path(tempfile.mkdtemp(prefix="audio-editor-tool-"))
SONGS: dict[str, dict] = {}
RESULTS: dict[str, dict] = {}
LOCK = threading.Lock()


def _peaks(audio: np.ndarray, buckets: int = 2400) -> list[list[float]]:
    mono = audio.mean(axis=0)
    n = max(1, len(mono) // buckets)
    m = mono[: n * (len(mono) // n)].reshape(-1, n)
    return np.round(np.stack([m.min(axis=1), m.max(axis=1)], axis=1), 4).tolist()


def _local_beat(a, t: float) -> float:
    near = a.beats[np.abs(a.beats - t) < 10]
    if len(near) > 4:
        return float(np.polyfit(np.arange(len(near)), near, 1)[0])
    return 60 / a.tempo


def load_song(data: bytes, filename: str) -> dict:
    from .analysis import analyze, sections

    sid = uuid.uuid4().hex[:10]
    safe = Path(filename).name or "song"
    path = WORKDIR / f"{sid}_{safe}"
    path.write_bytes(data)
    audio, sr = audio_io.load(path)
    a = analyze(audio, sr)
    secs = sections(a)
    SONGS[sid] = {"path": path, "name": safe, "audio": audio, "sr": sr, "a": a, "secs": secs}
    return {
        "id": sid,
        "name": safe,
        "duration": a.duration,
        "tempo": a.tempo,
        "beat": _local_beat(a, a.duration / 2),
        "bars": [float(x) for x in np.r_[a.bar_starts, a.bar_ends[-1]]],
        "beats": [float(x) for x in a.beats],
        "sections": [
            {"start": float(a.bar_starts[s]), "end": float(a.bar_ends[e]), "label": lab,
             "energy": float(a.score[s:e + 1].mean())}
            for s, e, lab in secs
        ],
        "peaks": _peaks(audio),
    }


def suggest(req: dict) -> dict:
    from .planner import Settings, plan
    from .render import segment_times

    song = SONGS[req["id"]]
    a = song["a"]
    cfg = Settings(target=float(req["length"]), keep=[tuple(k) for k in req.get("keep", [])],
                   start=req.get("start", "intro"), end=req.get("end", "auto"),
                   xfade_s=0.5 * 60 / a.tempo)
    p = plan(a, cfg, section_starts={s for s, _, _ in song["secs"]})
    return {"clips": [[float(t0), float(t1)] for t0, t1 in segment_times(a, p)],
            "natural_end": p.natural_end}


def render(req: dict) -> dict:
    from .render import Transition, overlap_len, render_parts, snap

    song = SONGS[req["id"]]
    a, audio, sr = song["a"], song["audio"], song["sr"]
    snap_mode = req.get("snap", "bar")
    times = []
    for t0, t1 in req["clips"]:
        t0 = 0.0 if t0 <= 0.05 else snap(a, float(t0), snap_mode)
        t1 = a.duration if t1 >= a.duration - 0.05 else snap(a, float(t1), snap_mode)
        if t1 - t0 < 0.2:
            raise ValueError("A clip is too short.")
        times.append((t0, t1))
    trans = []
    for (_, t1), j in zip(times[:-1], req.get("joins", [])):
        beat = _local_beat(a, t1)
        beats = float(j.get("beats", 0.5))
        x = overlap_len(a, t1, beats) if j.get("blend") == "overlay" else beats * beat
        trans.append(Transition(blend=j.get("blend", "crossfade"), xfade_s=max(0.005, x),
                                position=j.get("position", "before"),
                                hold=float(j.get("hold", 0.5)), fx=list(j.get("fx", [])),
                                fx_level=float(j.get("fx_level", 1.0))))
    while len(trans) < len(times) - 1:
        trans.append(Transition(fx=["echo", "swell"]))
    ending = req.get("ending", "fade")
    natural = ending == "natural" and times[-1][1] >= a.duration - 0.05
    out, cuts = render_parts(audio, sr, times, trans, natural_end=natural,
                             fade_out_s=float(req.get("fade_out", 4.0)),
                             beat_s=_local_beat(a, times[0][1]),
                             fade_in_s=float(req.get("fade_in", 0.0)),
                             gains_db=[float(x) for x in req.get("gains", [])],
                             master_db=float(req.get("master_db", 0.0)))
    rid = uuid.uuid4().hex[:10]
    wav = WORKDIR / f"result_{rid}.wav"
    audio_io.save(wav, out, sr)
    RESULTS[rid] = {"wav": wav, "audio": out, "sr": sr, "name": Path(song["name"]).stem}
    return {"rid": rid, "duration": out.shape[1] / sr, "cuts": cuts, "peaks": _peaks(out),
            "clips": [[t0, t1] for t0, t1 in times], "natural_end": natural}


def export(rid: str, fmt: str) -> Path:
    r = RESULTS[rid]
    if fmt == "wav":
        return r["wav"]
    path = WORKDIR / f"{r['name']}_edit_{rid}.{fmt}"
    if not path.exists():
        audio_io.save(path, r["audio"], r["sr"])
    return path


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK, headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            html = resources.files("audio_editor_tool").joinpath("web/index.html").read_bytes()
            return self._send(html, "text/html; charset=utf-8")
        if url.path.startswith("/api/result/"):
            rid = url.path.rsplit("/", 1)[-1]
            fmt = parse_qs(url.query).get("format", ["wav"])[0]
            if rid not in RESULTS or fmt not in ("wav", "mp3", "flac"):
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            try:
                path = export(rid, fmt)
            except Exception as e:  # e.g. ffmpeg missing for mp3
                return self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            ctype = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}[fmt]
            headers = {}
            if "download" in parse_qs(url.query):
                headers["Content-Disposition"] = (
                    f'attachment; filename="{RESULTS[rid]["name"]}_edit.{fmt}"')
            return self._send(path.read_bytes(), ctype, headers=headers)
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        url = urlparse(self.path)
        size = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(size)
        try:
            if url.path == "/api/load":
                name = unquote(self.headers.get("X-Filename", "song"))
                with LOCK:
                    return self._json(load_song(body, name))
            req = json.loads(body or b"{}")
            if req.get("id") not in SONGS:
                return self._json({"error": "Open a song first."}, HTTPStatus.BAD_REQUEST)
            if url.path == "/api/suggest":
                with LOCK:
                    return self._json(suggest(req))
            if url.path == "/api/render":
                with LOCK:
                    return self._json(render(req))
        except Exception as e:
            return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def _open_window(url: str) -> None:
    """Open as a standalone app window (Chrome/Edge app mode) if possible, else a browser tab."""
    candidates = [
        shutil.which(n) for n in ("google-chrome", "chrome", "chromium", "chromium-browser",
                                  "msedge", "microsoft-edge")
    ]
    if sys.platform == "darwin":
        candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                       "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    elif os.name == "nt":
        for base in (os.environ.get("PROGRAMFILES", ""), os.environ.get("PROGRAMFILES(X86)", ""),
                     os.environ.get("LOCALAPPDATA", "")):
            candidates += [os.path.join(base, r"Google\Chrome\Application\chrome.exe"),
                           os.path.join(base, r"Microsoft\Edge\Application\msedge.exe")]
    for exe in candidates:
        if exe and os.path.exists(exe):
            try:
                subprocess.Popen([exe, f"--app={url}", "--window-size=1400,900"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                continue
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="audio-editor-tool-gui", description="Open the visual editor.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-window", action="store_true", help="don't open a window, just serve")
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Audio Editor Tool running at {url}  (Ctrl+C to quit)")
    if not args.no_window:
        threading.Timer(0.5, _open_window, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(WORKDIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
