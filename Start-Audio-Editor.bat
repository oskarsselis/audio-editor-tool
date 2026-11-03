@echo off
REM Double-click to open the Audio Editor Tool window.
cd /d "%~dp0"
where uv >nul 2>nul || (
  echo uv is not installed. See "Windows setup" in README.md.
  pause
  exit /b 1
)
uv run audio-editor-tool-gui
if errorlevel 1 pause
