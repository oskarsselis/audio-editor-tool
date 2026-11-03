# Audio Editor Tool

Audio Editor Tool makes a short edit of a song. You select clips of the song. The tool joins the clips on the beat with smooth transitions and transition effects.

![Editor window](docs/screenshot.png)

## Requirements

- [uv](https://docs.astral.sh/uv/). uv also installs Python.
- [Git](https://git-scm.com/).
- [ffmpeg](https://ffmpeg.org/). You must have ffmpeg to open and save MP3 and M4A files. WAV, FLAC, and OGG files do not need ffmpeg.

## Install on Windows

1. Open PowerShell.
2. Install uv:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. Install Git:

   ```powershell
   winget install -e --id Git.Git
   ```

4. Install ffmpeg:

   ```powershell
   winget install -e --id Gyan.FFmpeg
   ```

5. Close PowerShell. Then open a new PowerShell window.
6. Download the tool and its dependencies:

   ```powershell
   cd $HOME\Documents
   git clone https://github.com/oskarsselis/audio-editor-tool.git
   cd audio-editor-tool
   uv sync
   ```

## Install on macOS and Linux

1. Install uv, Git, and ffmpeg with the package manager of your system.
2. Download the tool and its dependencies:

   ```bash
   git clone https://github.com/oskarsselis/audio-editor-tool.git
   cd audio-editor-tool
   uv sync
   ```

## Start the editor

- On Windows, double-click `Start-Audio-Editor.bat` in the tool folder.
- On all systems, run this command in the tool folder:

  ```bash
  uv run audio-editor-tool-gui
  ```

The editor opens in a new window. Keep the console window open while you use the editor. To stop the editor, close the console window.

## Make an edit

1. Click **Open audio** and select a song. The tool finds the beats, the bars, and the sections of the song.
2. Add clips. Drag on the waveform, or click **+ Add clip**. A new clip starts at the playhead.
3. Adjust each clip. Drag the clip edges, or type the start and end times.
4. Set the transition between each pair of clips.
5. Set the fades and the master volume.
6. Click **Render preview**. Listen to the result.
7. Select the output folder and the file format. Click **Save**.

## Editor settings

| Setting | Function |
|---|---|
| **Snap** | Moves the clip edges to the nearest bar line, to the nearest beat, or to no grid (**Off**). |
| **Sections** | The letters above the waveform show the song sections. The same letter shows similar music. |
| **Clip buttons** | ▶ plays the clip one time. ⟳ plays the clip again and again. ↑ and ↓ change the play order. **+** adds a clip after the clip. ✕ removes the clip. |
| **Vol** | Changes the volume of one clip, in dB. 0 dB is the original volume. |
| **Transition preset** | Sets the blend and the effects with one selection. |
| **Blend** | **Crossfade**: a short fade into the first beat of the next clip. **DJ bass swap**: the high frequencies blend, and the bass changes on the beat. **Overlay**: the next clip starts under the end of the previous clip. |
| **Length** | The length of the blend, in beats. 4 beats are 1 bar. |
| **Effects** | **Riser**: a noise sound that increases into the next clip. **Swell**: a reverse reverb into the next clip. **Echo**: echoes of the last beat of the previous clip. **Sweep**: removes the bass before the transition. |
| **FX vol** | The volume of the effects, in percent. 100 % is the normal volume. |
| **Auto-pick** | Selects clips for a target length. The tool keeps the clips that are in the list. |
| **Fade in**, **Fade out** | Select the checkbox to fade the start or the end of the output. Type the fade length in seconds. If **Fade out** is off, the output stops at the end of the last clip. |
| **Master volume** | Changes the volume of the full output, in dB. A limiter prevents clipping. |
| **⟳** (result) | Plays the result again and again. |
| **Save to** | The output folder. Click **Browse** to select a folder, or type the path. The tool does not overwrite a file. It adds a number to the new file name. |

## Command line

The command line makes an edit without the editor window.

Make a 90-second edit. The tool selects the clips:

```bash
uv run audio-editor-tool song.mp3 -l 1:30
```

Make an edit from two clips, with echo and swell effects and a fade-out:

```bash
uv run audio-editor-tool song.mp3 -p 0:30-1:00,2:00-2:30 --fx echo,swell --end fade
```

To see all the options, run `uv run audio-editor-tool --help`.

## Troubleshooting

| Problem | Solution |
|---|---|
| PowerShell does not find `uv`, `git`, or `ffmpeg`. | Open a new PowerShell window. If the problem continues, restart the computer. |
| PowerShell does not find `winget`. | Install **App Installer** from the Microsoft Store. |
| The editor does not start because the port is in use. | Close the other editor window. Or run `uv run audio-editor-tool-gui --port 8766`. |
| **Browse** does not open a folder dialog. | Type the folder path in **Save to**. |
| The tool cannot save an MP3 file. | Install ffmpeg. Then start the editor again. |

## Update

Run these commands in the tool folder:

```bash
git pull
uv sync
```

## Tests

```bash
uv run pytest
```

## Limits

- The tool gives the best results with a constant tempo and 4/4 time.
- The tool does not recognize lyrics. Listen to the preview to find cut vocal lines.
- All clips in one edit come from one song.

## License

MIT
