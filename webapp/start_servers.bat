@echo off
REM Start both backend and frontend servers
REM Paths are quoted and derived from this script so "WINDOWS 11" etc. work.

setlocal enabledelayedexpansion

echo Starting Contrastive Fruits servers...
echo.

REM Script is in webapp\; parent is repo root (or webapp parent)
set "WEBAPP_DIR=%~dp0"
set "WEBAPP_DIR=%WEBAPP_DIR:~0,-1%"
for %%I in ("%WEBAPP_DIR%") do set "WEBAPP_DIR=%%~fI"
set "REPO_ROOT=%WEBAPP_DIR%"
set "CKPT=%REPO_ROOT%\ckpt_epoch_1000\ckpt_epoch_1000.pt"
set "GALLERY=%REPO_ROOT%\gallery"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
set "PY_SCRIPT=%REPO_ROOT%\scripts\scripts\python_model_service.py"
set "FRONTEND_DIR=%REPO_ROOT%\frontend"

REM If running from repo root, webapp is subdir
if not exist "%CKPT%" (
    set "REPO_ROOT=%~dp0.."
    for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
    set "CKPT=%REPO_ROOT%\webapp\ckpt_epoch_1000\ckpt_epoch_1000.pt"
    set "GALLERY=%REPO_ROOT%\webapp\gallery"
    set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
    if not exist "%VENV_PYTHON%" set "VENV_PYTHON=%REPO_ROOT%\webapp\.venv\Scripts\python.exe"
    set "PY_SCRIPT=%REPO_ROOT%\scripts\python_model_service.py"
    set "FRONTEND_DIR=%REPO_ROOT%\webapp\frontend"
)

echo [1/2] Starting Flask backend (port 8001)...
start "Flask Backend" cmd /c "cd /d \"%REPO_ROOT%\" ^& \"%VENV_PYTHON%\" \"%PY_SCRIPT%\" --ckpt \"%CKPT%\" --gallery-root \"%GALLERY%\" --port 8001 ^& pause"

timeout /t 3 /nobreak

echo [2/2] Starting Frontend (port 8080)...
start "Frontend" cmd /c "cd /d \"%FRONTEND_DIR%\" ^& npm start"

echo.
echo All servers started!
echo Open browser to: http://127.0.0.1:8080/main.html
echo.
pause
