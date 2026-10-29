@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if not defined SOPG_PY set "SOPG_PY=%~dp0.venv\Scripts\python.exe"
set "PY=%SOPG_PY%"
if not exist "%PY%" (
  echo Python venv not found: %PY%
  echo See README.md "Install" to rebuild it, or set SOPG_PY to your python.exe.
  pause
  exit /b 1
)
if not exist "models\hand_landmarker.task" (
  echo models\hand_landmarker.task is missing. See models\README.md to download it.
  pause
  exit /b 1
)
echo Starting SOP-Guard UI (first start takes 10-20 seconds)...
"%PY%" main_ui.py
if errorlevel 1 pause
