@echo off
REM Contrastive Fruits - Complete Startup Script
REM Starts both Flask backend and Frontend HTTP server

setlocal enabledelayedexpansion

REM So "run_backend.bat" and "run_frontend.bat" are found (they live next to this file)
cd /d "%~dp0"

echo ========================================
echo  Contrastive Fruits Service Launcher
echo ========================================
echo.

REM Define paths (point to this Git repo)
set "WEBAPP_DIR=C:\Users\WINDOWS 11\Documents\GitHub\SEGP"
set "REPO_ROOT=%WEBAPP_DIR%"
set "CKPT=%REPO_ROOT%\webapp\ckpt_epoch_1000\ckpt_epoch_1000.pt"
set "GALLERY=%REPO_ROOT%\webapp\gallery"
set "FRONTEND_DIR=%REPO_ROOT%\webapp\frontend"

REM Prefer .venv in repo root; fallback to webapp\.venv
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" set "VENV_PYTHON=%REPO_ROOT%\webapp\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo ERROR: Python venv not found.
    echo Tried: %REPO_ROOT%\.venv
    echo        %REPO_ROOT%\webapp\.venv
    echo Create one with: cd "%REPO_ROOT%" ^& python -m venv .venv
    echo Then: .venv\Scripts\pip install -r contrastive-fruits\requirements.txt
    pause
    exit /b 1
)
echo Using Python: %VENV_PYTHON%

REM Check if checkpoint exists
if not exist "%CKPT%" (
    echo ERROR: Checkpoint not found at %CKPT%
    pause
    exit /b 1
)

REM Check if gallery exists
if not exist "%GALLERY%" (
    echo ERROR: Gallery not found at %GALLERY%
    echo Creating empty gallery...
    mkdir "%GALLERY%"
)

echo Starting services...
echo.

REM Start backend via helper (avoids passing paths with spaces to start)
echo [1/2] Starting Flask backend (port 8001)...
start "Contrastive Fruits - Flask Backend" cmd /c run_backend.bat

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak

REM Start frontend via helper (avoids passing paths with spaces to start)
echo [2/2] Starting Frontend (port 8080)...
start "Contrastive Fruits - Frontend" cmd /c run_frontend.bat

echo.
echo ========================================
echo All services started!
echo.
echo Backend:  http://127.0.0.1:8001/health
echo Frontend: http://127.0.0.1:8080/main.html
echo.
echo Close either window to stop that service.
echo ========================================
echo.
pause
