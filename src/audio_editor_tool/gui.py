"""Visual editor: a local web page shown in its own app window.

Run `audio-editor-tool-gui`. A small server on 127.0.0.1 does the audio work.
On Windows and macOS the page opens in a native window (pywebview); closing the
window stops the app. Elsewhere it opens in the browser, and the app stops soon
after the page is closed. Everything stays on your computer.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from . import audio_io

WORKDIR = Path(tempfile.mkdtemp(prefix="audio-editor-tool-"))
SONGS: OrderedDict[str, dict] = OrderedDict()     # most recent last
RESULTS: OrderedDict[str, dict] = OrderedDict()
MAX_SONGS, MAX_RESULTS = 3, 5                     # older ones are dropped to save memory
LOCK = threading.Lock()
WINDOW = None                                     # the pywebview window, if one is open
SERVER: ThreadingHTTPServer | None = None
APP_ID = "audio-editor-tool"

PROJECT_EXT = ".aetproj"
PROJECT_FORMAT = "audio-editor-tool-project"
AUDIO_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
               ".ogg": "audio/ogg", ".m4a": "audio/mp4"}


def _remember(store: OrderedDict, key: str, value: dict, limit: int) -> None:
    store[key] = value
    while len(store) > limit:
        _, old = store.popitem(last=False)
        for k in ("path", "wav"):
            if k in old:
                Path(old[k]).unlink(missing_ok=True)


def _peaks(audio: np.ndarray, buckets: int = 2400) -> list[list[float]]:
    mono = audio.mean(axis=0)
    n = max(1, len(mono) // buckets)
    m = mono[: n * (len(mono) // n)].reshape(-1, n)
    return np.round(np.stack([m.min(axis=1), m.max(axis=1)], axis=1), 4).tolist()


def _safe_name(name: str, fallback: str) -> str:
    stem = "".join(c for c in str(name or "") if c not in '<>:"/\\|?*\0').strip(" .")
    return stem or fallback


# ---------------------------------------------------------------- songs

def _add_song(data: bytes, filename: str, analysis_npz: bytes | None = None) -> dict:
    """Store and analyze a song. Returns what the page needs to draw and edit it."""
    from .analysis import Analysis, analyze, sections

    sid = uuid.uuid4().hex[:10]
    name = Path(filename).name or "song"
    path = WORKDIR / f"{sid}_{name}"
    path.write_bytes(data)
    audio, sr = audio_io.load(path)
    a = None
    if analysis_npz:
        try:
            a = Analysis.from_npz(analysis_npz)
        except Exception:                       # damaged or old project: analyze again
            a = None
    a = a or analyze(audio, sr)
    secs = sections(a)
    _remember(SONGS, sid, {"path": path, "name": name, "audio": audio, "sr": sr, "a": a,
                           "secs": secs}, MAX_SONGS)
    return {
        "id": sid,
        "name": name,
        "duration": a.duration,
        "beat": a.beat_period(a.duration / 2),
        "bars": [float(x) for x in np.r_[a.bar_starts, a.bar_ends[-1]]],
        "beats": [float(x) for x in a.beats],
        "sections": [{"start": float(a.bar_starts[s]), "end": float(a.bar_ends[e]),
                      "label": lab, "energy": float(a.score[s:e + 1].mean())}
                     for s, e, lab in secs],
        "peaks": _peaks(audio),
    }


def suggest(req: dict) -> dict:
    from .planner import Settings, plan
    from .render import segment_times

    song = SONGS[req["id"]]
    a = song["a"]
    cfg = Settings(target=float(req["length"]), keep=[tuple(k[:2]) for k in req.get("keep", [])],
                   start=req.get("start", "intro"), end=req.get("end", "auto"),
                   xfade_s=0.5 * a.beat_period(a.duration / 2))
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
        beats = float(j.get("beats", 0.5))
        x = overlap_len(a, t1, beats) if j.get("blend") == "overlay" else beats * a.beat_period(t1)
        trans.append(Transition(blend=j.get("blend", "crossfade"), xfade_s=max(0.005, x),
                                position=j.get("position", "before"),
                                hold=float(j.get("hold", 0.5)), fx=list(j.get("fx", [])),
                                fx_level=float(j.get("fx_level", 1.0))))
    while len(trans) < len(times) - 1:
        trans.append(Transition(fx=["echo", "swell"]))
    # Fade-out off: the output ends where the last clip ends.
    fade_out = float(req.get("fade_out", 4.0)) if req.get("fade_out_on", True) else 0.0
    fade_in = float(req.get("fade_in", 0.0)) if req.get("fade_in_on", True) else 0.0
    out, cuts = render_parts(audio, sr, times, trans, natural_end=fade_out <= 0,
                             fade_out_s=fade_out, beat_s=a.beat_period(times[0][1]),
                             fade_in_s=fade_in,
                             gains_db=[float(x) for x in req.get("gains", [])],
                             master_db=float(req.get("master_db", 0.0)))
    rid = uuid.uuid4().hex[:10]
    wav = WORKDIR / f"result_{rid}.wav"
    audio_io.save(wav, out, sr)
    _remember(RESULTS, rid, {"wav": wav, "audio": out, "sr": sr,
                             "name": Path(song["name"]).stem}, MAX_RESULTS)
    return {"rid": rid, "duration": out.shape[1] / sr, "cuts": cuts, "peaks": _peaks(out),
            "natural_end": fade_out <= 0}


# ---------------------------------------------------------------- file dialogs

TK_DIALOG = r"""
import sys, tkinter as tk
from tkinter import filedialog
kind, start, name = sys.argv[1], sys.argv[2], sys.argv[3]
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
if kind == "folder":
    out = filedialog.askdirectory(initialdir=start or None, mustexist=True,
                                  title="Choose the output folder")
else:
    out = filedialog.asksaveasfilename(initialdir=start or None, initialfile=name,
                                       defaultextension=".aetproj", title="Save project",
                                       filetypes=[("Audio Editor project", "*.aetproj")])
print(out or "")
"""


def _dialog(kind: str, start: str, name: str = "") -> str:
    """Show a native folder ('folder') or save-file ('save') dialog; '' if cancelled.

    In the app window: its own dialog (stays on top of the window). Otherwise a
    separate process with tkinter, then the system's own dialog.
    """
    if WINDOW is not None:
        import webview
        if kind == "folder":
            r = WINDOW.create_file_dialog(webview.FileDialog.FOLDER, directory=start)
        else:
            r = WINDOW.create_file_dialog(webview.FileDialog.SAVE, directory=start,
                                          save_filename=name,
                                          file_types=("Audio Editor project (*.aetproj)",))
        if isinstance(r, (list, tuple)):
            r = r[0] if r else ""
        return str(r or "")
    q = lambda v: v.replace("'", "''")          # PowerShell single-quoted string
    attempts = [[sys.executable, "-c", TK_DIALOG, kind, start, name]]
    if os.name == "nt":
        if kind == "folder":
            ps = ("$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
                  f"$d.Description = 'Choose the output folder'; $d.SelectedPath = '{q(start)}';"
                  "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }")
        else:
            ps = ("$d = New-Object System.Windows.Forms.SaveFileDialog;"
                  "$d.Filter = 'Audio Editor project (*.aetproj)|*.aetproj';"
                  f"$d.InitialDirectory = '{q(start)}'; $d.FileName = '{q(name)}';"
                  "if ($d.ShowDialog() -eq 'OK') { $d.FileName }")
        attempts.append(["powershell", "-NoProfile", "-STA", "-Command",
                         "Add-Type -AssemblyName System.Windows.Forms;" + ps])
    elif sys.platform == "darwin":
        script = ('POSIX path of (choose folder with prompt "Choose the output folder")'
                  if kind == "folder" else
                  f'POSIX path of (choose file name with prompt "Save project" '
                  f'default name "{name}")')
        attempts.append(["osascript", "-e", script])
    for cmd in attempts:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            return r.stdout.strip()
        if cmd[0] == "osascript" and "-128" in r.stderr:
            return ""                            # user cancelled
    raise RuntimeError("Could not open a file dialog.")


def default_folder() -> str:
    home = Path.home()
    for name in ("Music", "Downloads", "Documents"):
        if (home / name).is_dir():
            return str(home / name)
    return str(home)


def browse(req: dict) -> dict:
    try:
        return {"folder": _dialog("folder", req.get("folder") or default_folder())}
    except RuntimeError:
        raise RuntimeError("Could not open the folder dialog. Type the folder path instead.")


def _unique(path: Path) -> Path:
    """'name.ext', or 'name (2).ext', ... so an existing file is never overwritten."""
    n, out = 2, path
    while out.exists():
        out = path.with_name(f"{path.stem} ({n}){path.suffix}")
        n += 1
    return out


def save_output(req: dict) -> dict:
    """Save the rendered result to a folder the user chose."""
    rid, fmt = req.get("rid"), req.get("format", "mp3")
    if rid not in RESULTS:
        raise ValueError("Render a preview first.")
    if fmt not in ("mp3", "wav", "flac"):
        raise ValueError(f"Unknown format: {fmt}")
    folder = Path(os.path.expanduser(str(req.get("folder") or default_folder())))
    if not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")
    r = RESULTS[rid]
    path = _unique(folder / f"{_safe_name(req.get('name'), r['name'] + '_edit')}.{fmt}")
    audio_io.save(path, r["audio"], r["sr"])
    return {"path": str(path)}


# ---------------------------------------------------------------- projects
# A project is one zip file: project.json (clips and settings), the song itself,
# and the analysis (so opening is instant).

def save_project(req: dict) -> dict:
    song = SONGS.get(req.get("id"))
    if not song:
        raise ValueError("Open a song first.")
    path = req.get("path") if not req.get("save_as") else None
    if not path:
        state = req.get("state", {})
        folder = (state.get("output") or {}).get("folder") or default_folder()
        name = Path(song["name"]).stem + PROJECT_EXT
        try:
            path = _dialog("save", folder, name)
        except RuntimeError:                     # no dialog: save next to the output
            path = str(_unique(Path(os.path.expanduser(folder)) / name))
        if not path:
            return {"path": ""}                  # cancelled
    path = Path(path)
    if path.suffix.lower() != PROJECT_EXT:
        path = path.with_name(path.name + PROJECT_EXT)
    meta = {"format": PROJECT_FORMAT, "version": 1, "audio": song["name"],
            "state": req.get("state", {})}
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as z:
        z.writestr("project.json", json.dumps(meta, indent=2))
        z.write(song["path"], "audio/" + song["name"], compress_type=zipfile.ZIP_STORED)
        z.writestr("analysis.npz", song["a"].to_npz())
    os.replace(tmp, path)                        # never leave a half-written project
    return {"path": str(path)}


def open_project(data: bytes) -> dict:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        meta = json.loads(z.read("project.json"))
    except (zipfile.BadZipFile, KeyError, ValueError):
        raise ValueError("This is not an Audio Editor project file.")
    if meta.get("format") != PROJECT_FORMAT:
        raise ValueError("This is not an Audio Editor project file.")
    name = Path(str(meta.get("audio", ""))).name
    try:
        audio = z.read("audio/" + name)
    except KeyError:
        raise ValueError("The project file does not contain its song.")
    npz = z.read("analysis.npz") if "analysis.npz" in z.namelist() else None
    song = _add_song(audio, name, npz)
    return {"song": song, "state": meta.get("state", {})}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = resources.files("audio_editor_tool").joinpath("web/index.html").read_bytes()
            return self._send(html, "text/html; charset=utf-8")
        if path == "/api/config":
            return self._json({"app": APP_ID, "folder": default_folder()})
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["api", "result"] and parts[2] in RESULTS:
            return self._send(RESULTS[parts[2]]["wav"].read_bytes(), "audio/wav")
        if len(parts) == 4 and parts[:2] == ["api", "song"] and parts[3] == "audio" \
                and parts[2] in SONGS:
            song = SONGS[parts[2]]
            ctype = AUDIO_TYPES.get(Path(song["name"]).suffix.lower(), "application/octet-stream")
            return self._send(song["path"].read_bytes(), ctype)
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if path == "/api/quit":                         # stop the app (used by tests)
            self._json({})
            LIFE.quit = True
            if WINDOW is not None:
                WINDOW.destroy()
            elif SERVER is not None:
                threading.Thread(target=SERVER.shutdown, daemon=True).start()
            return
        if path in ("/api/ping", "/api/bye"):
            LIFE.ping() if path == "/api/ping" else LIFE.bye()
            return self._json({})
        try:
            if path == "/api/load":
                with LOCK:
                    return self._json(_add_song(body, unquote(self.headers.get("X-Filename", "song"))))
            if path == "/api/project/open":
                with LOCK:
                    return self._json(open_project(body))
            req = json.loads(body or b"{}")
            handlers = {"/api/browse": browse, "/api/save": save_output,
                        "/api/project/save": save_project}
            if path in handlers:
                return self._json(handlers[path](req))
            if req.get("id") not in SONGS:
                return self._json({"error": "Open a song first."}, HTTPStatus.BAD_REQUEST)
            if path in ("/api/suggest", "/api/render"):
                with LOCK:
                    return self._json((suggest if path == "/api/suggest" else render)(req))
        except Exception as e:
            return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


class _Life:
    """Browser mode only: stop the app after its page is closed.

    The page pings every few seconds and says "bye" when it closes. A reload says
    bye and pings again at once, so it is not mistaken for a close. A minimized
    window can slow the pings to one per minute, so silence alone waits 3 minutes.
    """

    def __init__(self):
        self.quit = False
        self.last_ping = self.bye_at = 0.0
        self.started = time.monotonic()

    def ping(self):
        self.last_ping = time.monotonic()

    def bye(self):
        self.bye_at = time.monotonic()

    def should_stop(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if self.quit:
            return True
        if not self.last_ping:                          # page never opened
            return now - self.started > 120
        if self.bye_at > self.last_ping and now - self.bye_at > 5:
            return True
        return now - self.last_ping > 180


LIFE = _Life()


def _browser_window(url: str) -> None:
    """Open as a standalone app window (Chrome/Edge app mode) if possible, else a browser tab."""
    candidates = [shutil.which(n) for n in ("google-chrome", "chrome", "chromium",
                                            "chromium-browser", "msedge", "microsoft-edge")]
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


def _native_window(url: str, self_test: bool = False) -> bool | None:
    """Show the editor in a native window and block until it is closed.

    Returns None if no native window is possible here (then use the browser).
    With self_test, check that the page loads and the header is visible, then
    close the window; returns True if the check passed.
    """
    global WINDOW
    try:
        import webview
    except ImportError:
        return None
    result = {"ok": not self_test}

    def check(win):
        if not win.events.loaded.wait(60):
            print("self-test: page did not load", flush=True)
        else:
            info = win.evaluate_js("JSON.stringify({top: document.querySelector('#openBtn')"
                                   ".getBoundingClientRect().top, title: document.title})")
            print("self-test:", info, flush=True)
            data = json.loads(info or "{}")
            result["ok"] = data.get("title") == "Audio Editor Tool" and data.get("top", -1) >= 0
        win.destroy()

    storage = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / APP_ID
    try:
        WINDOW = webview.create_window("Audio Editor Tool", url, width=1400, height=900,
                                       min_size=(960, 640), background_color="#0f1115",
                                       text_select=True)
        webview.start(check if self_test else None, (WINDOW,) if self_test else None,
                      private_mode=False, storage_path=str(storage),
                      gui="edgechromium" if os.name == "nt" else None)
    except Exception as e:                              # e.g. WebView2 missing
        print(f"No app window ({e}); using the browser.", file=sys.stderr)
        WINDOW = None
        return None
    WINDOW = None
    return result["ok"]


def _detach(argv: list[str]) -> int:
    """Start the editor again as a hidden background process and return at once,
    so the console window that launched it can close."""
    cmd = [sys.executable, "-m", "audio_editor_tool.gui", *[a for a in argv if a != "--detach"]]
    kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL, "close_fds": True}
    if os.name != "nt":
        subprocess.Popen(cmd, start_new_session=True, **kw)
        return 0
    # A hidden console: tools started later (ffmpeg, dialogs) do not flash a window.
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    try:   # leave the launcher's job object, so closing the launcher does not stop us
        subprocess.Popen(cmd, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
    except OSError:                                     # breakaway not allowed
        subprocess.Popen(cmd, creationflags=flags, **kw)
    return 0


def _fatal(msg: str) -> None:
    print(msg, file=sys.stderr)
    if os.name == "nt":                                 # visible even without a console
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, msg, "Audio Editor Tool", 0x10)


def _make_server(port: int) -> ThreadingHTTPServer:
    try:
        return ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:                                     # port in use: take any free port
        return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="audio-editor-tool-gui", description="Open the visual editor.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--browser", action="store_true", help="use the web browser, not an app window")
    ap.add_argument("--no-window", action="store_true", help="only run the server (for tests)")
    ap.add_argument("--detach", action="store_true", help="run in the background without a console")
    ap.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.detach:
        return _detach(argv)

    global SERVER
    try:
        server = SERVER = _make_server(args.port)
    except OSError as e:
        _fatal(f"Could not start the editor: {e}")
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Audio Editor Tool running at {url}  (close the window to quit)", flush=True)
    code = 0
    try:
        if args.no_window:
            server.serve_forever()
        else:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            ok = None if args.browser else _native_window(url, args.self_test)
            if ok is None:                              # browser mode
                if args.self_test:
                    print("self-test: no app window on this system", flush=True)
                    return 0
                _browser_window(url)
                while not LIFE.should_stop():
                    time.sleep(1)
            elif not ok:
                code = 1
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(WORKDIR, ignore_errors=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
