@echo off
REM Run Flask backend from repo root. Double-click this from the webapp folder.
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%\webapp\.venv\Scripts\python.exe"
set "CKPT=%ROOT%\webapp\ckpt_epoch_1000\ckpt_epoch_1000.pt"
set "GALLERY=%ROOT%\webapp\gallery"
set "SCRIPT=%ROOT%\scripts\python_model_service.py"

cd /d "%ROOT%"
"%PY%" "%SCRIPT%" --ckpt "%CKPT%" --gallery-root "%GALLERY%" --port 8001
pause
