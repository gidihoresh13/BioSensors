@echo off
REM ---------------------------------------------------------------------------
REM Launch the detection GUI with Python 3.11 (the interpreter that has all the
REM required packages). Double-click this file, or run it from a terminal.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
py -3.11 gui_detect.py
if errorlevel 1 (
  echo.
  echo The GUI exited with an error. If it says a module is missing, install deps:
  echo     py -3.11 -m pip install -r requirements.txt
  pause
)
