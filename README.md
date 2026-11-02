# song-shortener

Make a shorter edit of a song: give a target length and (optionally) the parts that must stay. The tool finds bars and phrases, picks cut points where the music lines up, and joins them with beat-aligned crossfades, so the edit sounds like it was meant that way.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. Install `ffmpeg` to read and write MP3, M4A and similar formats. WAV, FLAC and OGG work without it.

```bash
uv sync
```

## Usage

```bash
# ~1:30 version
uv run song-shortener song.mp3 -l 1:30

# Must include 1:05–1:40 (e.g. the guitar solo) and leave out 2:10–2:30
uv run song-shortener song.mp3 -l 2:00 -k 1:05-1:40 -s 2:10-2:30 -o edit.mp3

# See the song's sections first, to help choose --keep ranges
uv run song-shortener song.mp3 --analyze
```

Example output:

```
Edit (3 parts):
    0:00.0  plays original 0:00.0 - 0:07.9  -> transition match 89%
    0:07.6  plays original 0:15.7 - 0:55.1  -> transition match 96%
    0:46.7  plays original 1:26.6 - 1:42.3
  ending: 4s fade-out
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `-l, --length` | – | Target length (`90`, `1:30`). Approximate: cuts stay on whole phrases. |
| `-k, --keep START-END` | – | Part that must be included. Repeatable. |
| `-s, --skip START-END` | – | Part that must be left out. Repeatable. |
| `-o, --output` | `<input>_short.<ext>` | Output file. |
| `--start intro\|any` | `intro` | `any` lets the edit start later, at a section start. |
| `--end auto\|outro\|fade` | `auto` | Keep the real ending, fade out, or let the tool choose. |
| `-p, --parts A-B,C-D` | – | Choose the parts yourself instead of `-l`. Edges snap to bars. |
| `--snap bar\|beat\|none` | `bar` | Snapping for `--parts` edges. |
| `--xfade BEATS` | `0.5` | Crossfade length in beats (4 = one bar in 4/4). |
| `--blend crossfade\|eq\|overlay` | `crossfade` | `eq` is a DJ-style bass swap: mids and highs blend, and the bass hands over on the downbeat. `overlay` starts the next part at full volume underneath the end of the previous one, and `--xfade` sets the overlap. |
| `--hold 0-1` | `0.5` | For overlay: the share of the overlap where the previous part stays at full volume before fading. |
| `--blend-pos before\|center\|after` | `before` | Where the blend sits: `before` lets the new part's downbeat hit at full volume, and `after` lets the old part's tail (e.g. a held vocal) carry over. |
| `--fade-out SEC` | `4` | Fade-out length when the real ending isn't used. |
| `--min-segment BARS` | `4` | Shortest piece between two cuts. |
| `--tolerance SEC` | `4` | How far the length may drift from the target. |
| `--analyze` | – | Print only the section map. |
| `--plan-json FILE` | – | Save the chosen cuts as JSON. |

## How it works

1. **Beats and bars:** librosa beat tracking. The downbeat is the beat with the strongest bass hits and chord changes. The bar grid is extended over quiet intros and outros.
2. **Bar similarity:** harmony (per-beat chroma), timbre (MFCC) and loudness for every bar.
3. **Planning:** dynamic programming over bars. Each jump from bar *i* to bar *j* is scored by how much bar *j* sounds like the bar it replaces (*i+1*). It prefers phrase-aligned skips (multiples of 4 or 8 bars), cutting whole sections, few cuts, and loud, repeated parts (choruses). Required parts are never skipped, and the length stays near the target.
4. **Rendering:** cuts snap to the drum hit. An equal-power crossfade runs just *before* the downbeat, so the new part's downbeat lands at full volume. Without the real ending, the edit fades out.

## Limitations

- Works best with a steady tempo and 4/4 time. Rubato, tempo changes or 3/4 time can give odd cuts.
- Lengths land on whole phrases, so the result can be a few seconds off the target. Use `--tolerance` to tighten or loosen this.
- Lyrics aren't understood. A cut can join two choruses with different words.

## Demo

```bash
uv run python examples/make_demo_song.py demo.wav   # synthetic song with known sections
uv run song-shortener demo.wav --analyze
uv run song-shortener demo.wav -l 60
```
