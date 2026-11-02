"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audio_io
from .analysis import analyze, sections
from .planner import Settings, plan
from .planner import jump_match
from .render import render, segment_times, snap


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
                    help="crossfade length in beats (default 0.5; 4 = one bar)")
    ap.add_argument("--blend", choices=["crossfade", "eq"], default="crossfade",
                    help="crossfade: plain equal-power; eq: DJ-style bass swap (cleaner long blends)")
    ap.add_argument("--blend-pos", choices=["before", "center", "after"], default="before",
                    help="blend before the cut (new downbeat at full volume), centered, or after "
                         "(old part's tail, e.g. a held vocal, carries over)")
    ap.add_argument("-p", "--parts", metavar="START-END,...",
                    help="choose the parts yourself instead of -l, e.g. 1:41-1:58,2:27-2:44")
    ap.add_argument("--snap", choices=["bar", "beat", "none"], default="bar",
                    help="with --parts: snap part edges to the nearest bar (default), beat, or not at all")
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

    if not args.analyze and args.length is None and not args.parts:
        ap.error("--length or --parts is required (or use --analyze)")

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

    if args.parts:
        return manual(args, audio, sr, a, beat)

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
    times = segment_times(a, p)
    return finish(args, audio, sr, a, beat, times, p.join_match, p.natural_end)


def manual(args, audio, sr, a, beat) -> int:
    times = []
    for chunk in args.parts.split(","):
        t0, t1 = parse_range(chunk)
        t0 = 0.0 if t0 <= 0.05 else snap(a, t0, args.snap)
        t1 = a.duration if t1 >= a.duration - 0.05 else snap(a, t1, args.snap)
        times.append((t0, t1))
    match = jump_match(a)
    joins = [float(match[a.bar_at(x1 - 0.05), a.bar_at(y0 + 0.05)])
             for (_, x1), (y0, _) in zip(times[:-1], times[1:])]
    natural = times[-1][1] >= a.duration - 0.05 and args.end != "fade"
    return finish(args, audio, sr, a, beat, times, joins, natural)


def finish(args, audio, sr, a, beat, times, joins, natural_end) -> int:
    xfade_s = args.xfade * beat
    out = render(audio, sr, times, natural_end, xfade_s, args.fade_out,
                 blend=args.blend, position=args.blend_pos)

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(
        f"{in_path.stem}_short{in_path.suffix if in_path.suffix.lower() != '.m4a' else '.wav'}")
    audio_io.save(out_path, out, sr)

    print(f"\nEdit ({len(times)} part{'s' if len(times) > 1 else ''}):")
    pos = 0.0
    for idx, (t0, t1) in enumerate(times):
        note = f"  -> transition match {100 * joins[idx]:.0f}%" if idx < len(joins) else ""
        print(f"  {fmt(pos):>7}  plays original {fmt(t0)} - {fmt(t1)}{note}")
        pos += t1 - t0
    if len(times) > 1:
        print(f"  blend: {args.blend}, {args.xfade:g} beats ({xfade_s:.2f}s), {args.blend_pos} the cut")
    print(f"  ending: {'original outro' if natural_end else f'{args.fade_out:g}s fade-out'}")
    target = f", target {fmt(args.length)}" if args.length else ""
    print(f"\nSaved {out_path}  ({fmt(out.shape[1] / sr)}{target})")

    if args.plan_json:
        Path(args.plan_json).write_text(json.dumps({
            "input": str(in_path), "output": str(out_path), "tempo": a.tempo,
            "parts": [{"from": round(t0, 3), "to": round(t1, 3)} for t0, t1 in times],
            "transition_match": [round(x, 3) for x in joins],
            "blend": args.blend, "xfade_beats": args.xfade, "blend_pos": args.blend_pos,
            "natural_end": natural_end, "duration": round(out.shape[1] / sr, 3),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
