"""End-to-end tests on a synthetic song with known structure (122 BPM, 2 s bars)."""

import importlib.util
import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from audio_editor_tool import audio_io
from audio_editor_tool.analysis import analyze, sections
from audio_editor_tool.cli import main as cli_main, parse_range, parse_time
from audio_editor_tool.planner import Settings, plan
from audio_editor_tool.render import Transition, render_parts, segment_times

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("demo", ROOT / "examples" / "make_demo_song.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = tmp_path_factory.mktemp("audio") / "demo.wav"
    mod.main(str(path))
    audio, sr = audio_io.load(path)
    return path, audio, sr, analyze(audio, sr)


def test_parse_helpers():
    assert parse_time("1:30") == 90
    assert parse_time("90.5") == 90.5
    assert parse_range("1:00-1:30") == (60, 90)


def test_analysis_finds_tempo_and_sections(demo):
    _, _, _, a = demo
    assert abs(a.bar_len - 4 * 60 / 122) < 0.05
    starts = [a.bar_starts[s] for s, _, _ in sections(a)]
    # verse 1 starts at 7.9 s, first chorus at 23.6 s
    assert any(abs(t - 7.9) < 0.3 for t in starts)
    assert any(abs(t - 23.6) < 0.3 for t in starts)


def test_plan_hits_target_and_keeps_parts(demo):
    _, _, _, a = demo
    keep = (72.0, 85.0)
    p = plan(a, Settings(target=75, keep=[keep]))
    times = segment_times(a, p)
    total = sum(t1 - t0 for t0, t1 in times)
    assert abs(total - 75) < 8
    assert any(t0 <= keep[0] + 2 and t1 >= keep[1] - 2 for t0, t1 in times)


def test_render_three_clips_with_effects(demo):
    _, audio, sr, a = demo
    times = [(23.6, 39.3), (55.1, 63.0), (86.6, 102.3)]
    trans = [Transition(fx=["echo", "swell"]), Transition(blend="overlay", xfade_s=a.bar_len)]
    out, cuts = render_parts(audio, sr, times, trans, natural_end=False, fade_out_s=3,
                             beat_s=a.bar_len / 4)
    expected = sum(t1 - t0 for t0, t1 in times) - a.bar_len
    assert abs(out.shape[1] / sr - expected) < 0.05
    assert len(cuts) == 2 and 0 < cuts[0] < cuts[1] < out.shape[1] / sr
    assert np.abs(out).max() <= 1.0
    assert np.abs(out[:, -100:]).max() < 0.01          # faded out


def test_cli_writes_output(demo, tmp_path):
    path, *_ = demo
    out = tmp_path / "short.wav"
    plan_json = tmp_path / "plan.json"
    assert cli_main([str(path), "-l", "60", "-o", str(out), "--fx", "echo,swell",
                     "--plan-json", str(plan_json)]) == 0
    audio, sr = audio_io.load(out)
    assert 50 < audio.shape[1] / sr < 70
    assert json.loads(plan_json.read_text())["parts"]


def test_gui_server_round_trip(demo):
    from http.server import ThreadingHTTPServer
    from audio_editor_tool import gui

    path, *_ = demo
    server = ThreadingHTTPServer(("127.0.0.1", 0), gui.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(url, data, headers=None):
        req = urllib.request.Request(base + url, data=data, headers=headers or {}, method="POST")
        return json.load(urllib.request.urlopen(req))

    try:
        assert b"Audio Editor Tool" in urllib.request.urlopen(base + "/").read()
        song = post("/api/load", path.read_bytes(), {"X-Filename": "demo.wav"})
        assert song["sections"] and song["peaks"]
        body = json.dumps({"id": song["id"], "clips": [[7.9, 23.6], [55.1, 70.8], [86.6, 102.3]],
                           "joins": [{"blend": "crossfade", "beats": 0.5, "fx": ["echo", "swell"]},
                                     {"blend": "eq", "beats": 4, "fx": ["riser"]}],
                           "gains": [0, -3, 2], "master_db": -1, "fade_in_on": True, "fade_in": 1,
                           "fade_out_on": True, "fade_out": 3}).encode()
        res = post("/api/render", body, {"Content-Type": "application/json"})
        assert len(res["cuts"]) == 2 and res["duration"] > 40
        wav = urllib.request.urlopen(f"{base}/api/result/{res['rid']}").read()
        assert wav[:4] == b"RIFF"
        # save into a chosen folder; a second save must not overwrite the first
        folder = path.parent / "out"
        folder.mkdir()
        body = {"rid": res["rid"], "folder": str(folder), "name": "my edit", "format": "wav"}
        first = post("/api/save", json.dumps(body).encode(), {"Content-Type": "application/json"})
        second = post("/api/save", json.dumps(body).encode(), {"Content-Type": "application/json"})
        assert Path(first["path"]).name == "my edit.wav"
        assert Path(second["path"]).name == "my edit (2).wav"
        # project: save to a given path, open it again -> same song, clips and settings
        state = {"clips": [[7.9, 23.6, -2], [55.1, 70.8, 0]], "master": -1,
                 "fade_out": {"on": False, "s": 4}}
        proj = folder / "my project.aetproj"
        saved = post("/api/project/save", json.dumps({"id": song["id"], "state": state,
                                                      "path": str(proj)}).encode(),
                     {"Content-Type": "application/json"})
        assert Path(saved["path"]) == proj and proj.stat().st_size > 100_000
        opened = post("/api/project/open", proj.read_bytes(), {"X-Filename": proj.name})
        assert opened["state"] == state
        assert opened["song"]["bars"] == song["bars"]          # analysis restored, not redone
        audio = urllib.request.urlopen(f"{base}/api/song/{opened['song']['id']}/audio").read()
        assert audio == path.read_bytes()
        try:
            post("/api/project/open", b"not a zip", {"X-Filename": "x.aetproj"})
            raise AssertionError("bad project accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 400
        if shutil.which("ffmpeg"):
            body["format"] = "mp3"
            mp3 = post("/api/save", json.dumps(body).encode(), {"Content-Type": "application/json"})
            assert Path(mp3["path"]).stat().st_size > 10_000
    finally:
        server.shutdown()


def test_clip_and_master_volume_and_fade_in(demo):
    _, audio, sr, a = demo
    times = [(23.6, 39.3), (55.1, 70.8)]
    trans = [Transition()]

    def run(**kw):
        out, cuts = render_parts(audio, sr, times, trans, natural_end=False, fade_out_s=2,
                                 beat_s=a.bar_len / 4, **kw)
        return out, cuts[0]

    def rms_db(x):
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)))

    base, cut = run()
    quiet, _ = run(gains_db=[-6, 0])
    mid1 = slice(int(2 * sr), int((cut - 1) * sr))          # inside clip 1
    mid2 = slice(int((cut + 1) * sr), int((cut + 8) * sr))  # inside clip 2
    assert abs((rms_db(quiet[:, mid1]) - rms_db(base[:, mid1])) + 6) < 0.3
    assert abs(rms_db(quiet[:, mid2]) - rms_db(base[:, mid2])) < 0.3

    softer, _ = run(master_db=-3)
    assert abs((rms_db(softer[:, mid1]) - rms_db(base[:, mid1])) + 3) < 0.3

    faded, _ = run(fade_in_s=2)
    assert np.abs(faded[:, :50]).max() < 0.01                      # silent at the start
    assert np.allclose(faded[:, int(3 * sr):int(4 * sr)], base[:, int(3 * sr):int(4 * sr)])


def test_cli_gain_and_fade_in(demo, tmp_path):
    path, *_ = demo
    out = tmp_path / "vol.wav"
    assert cli_main([str(path), "-p", "0:23.6-0:39.3,1:26.6-1:42.3", "--gains", "0,-3",
                     "--master-gain", "-1", "--fade-in", "1.5", "--end", "fade",
                     "-o", str(out)]) == 0
    audio, sr = audio_io.load(out)
    assert np.abs(audio[:, :50]).max() < 0.01


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_save_mp3(tmp_path):
    """MP3 export goes through a temporary WAV; this failed on Windows before."""
    sr = 44100
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    out = tmp_path / "tone.mp3"
    audio_io.save(out, np.stack([tone, tone]), sr)
    back, _ = audio_io.load(out)
    assert abs(back.shape[1] / sr - 1.0) < 0.1


def test_fade_out_off_ends_at_clip_end(demo):
    _, audio, sr, a = demo
    out, _ = render_parts(audio, sr, [(23.6, 39.3)], [], natural_end=True, fade_out_s=0)
    assert abs(out.shape[1] / sr - 15.7) < 0.01
    assert np.abs(out[:, -int(0.5 * sr):-int(0.1 * sr)]).max() > 0.05   # not faded


def test_analysis_round_trip(demo):
    from audio_editor_tool.analysis import Analysis
    *_, a = demo
    b = Analysis.from_npz(a.to_npz())
    assert b.duration == a.duration and b.tempo == a.tempo
    assert np.array_equal(b.bar_starts, a.bar_starts) and np.array_equal(b.sim, a.sim)
    assert b.beat_period(30) == a.beat_period(30)


def test_editor_keeps_only_recent_items(tmp_path):
    from collections import OrderedDict
    from audio_editor_tool.gui import _remember
    store = OrderedDict()
    files = []
    for i in range(5):
        f = tmp_path / f"r{i}.wav"
        f.write_bytes(b"x")
        files.append(f)
        _remember(store, str(i), {"wav": f}, limit=3)
    assert list(store) == ["2", "3", "4"]
    assert not files[0].exists() and files[4].exists()        # old temp files deleted


def test_browser_mode_stops_after_page_closes():
    from audio_editor_tool.gui import _Life
    life = _Life()
    t = life.started
    assert not life.should_stop(t + 60) and life.should_stop(t + 121)   # page never opened
    life.last_ping = t + 1
    assert not life.should_stop(t + 100)
    life.bye_at = t + 2                                                 # page closed
    assert not life.should_stop(t + 4) and life.should_stop(t + 8)
    life.last_ping = t + 3                                              # reload: pings again
    assert not life.should_stop(t + 30)
    assert life.should_stop(t + 3 + 181)                                # silent for 3 minutes


def test_detach_keeps_running_after_launcher_exits():
    """The Windows launcher starts the app with --detach and then closes."""
    import socket
    import subprocess
    import sys
    import time
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    launcher = subprocess.run([sys.executable, "-m", "audio_editor_tool.gui", "--detach",
                               "--no-window", "--port", str(port)], timeout=30)
    assert launcher.returncode == 0
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):                     # the hidden app needs a moment to start
        try:
            info = json.load(urllib.request.urlopen(url + "/api/config", timeout=1))
            break
        except OSError:
            time.sleep(0.5)
    else:
        raise AssertionError("detached app did not start")
    assert info["app"] == "audio-editor-tool"
    urllib.request.urlopen(urllib.request.Request(url + "/api/quit", data=b"", method="POST"))
    for _ in range(20):
        try:
            urllib.request.urlopen(url + "/api/config", timeout=1)
            time.sleep(0.5)
        except OSError:
            return
    raise AssertionError("app did not stop after quit")
