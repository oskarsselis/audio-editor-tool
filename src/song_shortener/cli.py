"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audio_io
from .analysis import analyze, sections
from .planner import Settings, plan
from .render import render


def parse_time(s: str) -> float:
    """'90', '1:30', '1:30.5' -> seconds."""
    parts = s.strip().split(":")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return sec


def parse_range(s: str) -> tuple[float, float]:
    a, b = s.split("-", 1)
    t0, t1 = parse_time(a), parse_time(b)
    if t1 <= t0:
        raise argparse.ArgumentTypeError(f"range end must be after start: {s}")
    return t0, t1


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="song-shortener",
        description="Make a shorter edit of a song with smooth, beat-matched transitions.",
    )
    ap.add_argument("input", help="audio file (mp3, wav, flac, m4a, ...)")
    ap.add_argument("-l", "--length", type=parse_time,
                    help="target length, e.g. 90 or 1:30 (approximate)")
    ap.add_argument("-o", "--output", help="output file (default: <input>_short.<ext>)")
    ap.add_argument("-k", "--keep", type=parse_range, action="append", default=[],
                    metavar="START-END", help="part that must be included, e.g. 1:05-1:40 (repeatable)")
    ap.add_argument("-s", "--skip", type=parse_range, action="append", default=[],
                    metavar="START-END", help="part that must be left out (repeatable)")
    ap.add_argument("--start", choices=["intro", "any"], default="intro",
                    help="intro: begin at the song start (default); any: may start later")
    ap.add_argument("--end", choices=["auto", "outro", "fade"], default="auto",
                    help="outro: keep the real ending; fade: fade out; auto: choose best (default)")
    ap.add_argument("--xfade", type=float, default=0.5, metavar="BEATS",
                    help="crossfade length in beats (default 0.5)")
    ap.add_argument("--fade-out", type=float, default=4.0, metavar="SEC",
                    help="fade-out length when not using the real ending (default 4)")
    ap.add_argument("--min-segment", type=int, default=4, metavar="BARS",
                    help="shortest piece kept between cuts (default 4 bars)")
    ap.add_argument("--tolerance", type=float, default=4.0, metavar="SEC",
                    help="how far the length may drift from target (default 4)")
    ap.add_argument("--analyze", action="store_true",
                    help="only print the song map (sections + times) to help pick --keep parts")
    ap.add_argument("--plan-json", metavar="FILE", help="also save the edit plan as JSON")
    args = ap.parse_args(argv)

    if not args.analyze and args.length is None:
        ap.error("--length is required (or use --analyze)")

    audio, sr = audio_io.load(args.input)
    print(f"Analyzing {args.input} ...", file=sys.stderr)
    a = analyze(audio, sr)
    secs = sections(a)
    beat = 60 / a.tempo
    print(f"Length {fmt(a.duration)} | tempo ~{a.tempo:.0f} BPM | {a.n} bars of ~{a.bar_len:.2f}s")

    if args.analyze:
        print("\nSong map (same letter = similar music; energy 0-9):")
        for s, e, lab in secs:
            en = int(round(9 * a.score[s:e + 1].mean()))
            print(f"  {fmt(a.bar_starts[s]):>7} - {fmt(a.bar_ends[e]):>7}  [{lab}]  "
                  f"bars {s + 1:>3}-{e + 1:<3} energy {'#' * en}")
        return 0

    if args.length >= a.duration:
        print("Target length is not shorter than the song.", file=sys.stderr)
        return 1

    kept = sum(t1 - t0 for t0, t1 in args.keep)
    if kept > args.length:
        print(f"Warning: --keep parts total {fmt(kept)}, longer than the target; "
              "the edit will be longer.", file=sys.stderr)

    cfg = Settings(target=args.length, keep=args.keep, skip=args.skip, start=args.start,
                   end=args.end, min_segment=args.min_segment, tolerance=args.tolerance,
                   xfade_s=args.xfade * beat)
    p = plan(a, cfg, section_starts={s for s, _, _ in secs})
    out, times = render(audio, sr, a, p, cfg.xfade_s, args.fade_out)

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(
        f"{in_path.stem}_short{in_path.suffix if in_path.suffix.lower() != '.m4a' else '.wav'}")
    audio_io.save(out_path, out, sr)

    print(f"\nEdit ({len(times)} part{'s' if len(times) > 1 else ''}):")
    pos = 0.0
    for idx, (t0, t1) in enumerate(times):
        note = ""
        if idx < len(p.join_match):
            note = f"  -> transition match {100 * p.join_match[idx]:.0f}%"
        print(f"  {fmt(pos):>7}  plays original {fmt(t0)} - {fmt(t1)}{note}")
        pos += t1 - t0 - (cfg.xfade_s if idx < len(times) - 1 else 0)
    print(f"  ending: {'original outro' if p.natural_end else f'{args.fade_out:.0f}s fade-out'}")
    print(f"\nSaved {out_path}  ({fmt(out.shape[1] / sr)}, target {fmt(args.length)})")

    if args.plan_json:
        Path(args.plan_json).write_text(json.dumps({
            "input": str(in_path), "output": str(out_path), "tempo": a.tempo,
            "parts": [{"from": round(t0, 3), "to": round(t1, 3)} for t0, t1 in times],
            "transition_match": [round(x, 3) for x in p.join_match],
            "natural_end": p.natural_end, "duration": round(out.shape[1] / sr, 3),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
