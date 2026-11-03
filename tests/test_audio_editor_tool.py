"""End-to-end tests on a synthetic song with known structure (122 BPM, 2 s bars)."""

import importlib.util
import json
import threading
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
                           "ending": "fade"}).encode()
        res = post("/api/render", body, {"Content-Type": "application/json"})
        assert len(res["cuts"]) == 2 and res["duration"] > 40
        wav = urllib.request.urlopen(f"{base}/api/result/{res['rid']}?format=wav").read()
        assert wav[:4] == b"RIFF"
    finally:
        server.shutdown()
