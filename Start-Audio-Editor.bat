@echo off
REM Double-click to open Audio Editor Tool. The console closes by itself.
cd /d "%~dp0"
where uv >nul 2>nul || (
  echo uv is not installed. See "Install on Windows" in README.md.
  pause
  exit /b 1
)
uv sync -q || (pause & exit /b 1)
uv run --no-sync audio-editor-tool-gui --detach || pause
