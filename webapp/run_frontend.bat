@echo off
REM Run frontend server. Double-click this from the webapp folder.
setlocal
set "FRONTEND=%~dp0frontend"
cd /d "%FRONTEND%"
npm start
pause
