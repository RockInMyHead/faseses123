@echo off
echo FACE CLUSTERING SERVER STARTUP
echo ===============================
echo.
echo Current directory: %CD%
echo Script location: %~dp0
echo.
echo Starting server...
echo.
cd /d "%~dp0"
python main.py
echo.
pause
