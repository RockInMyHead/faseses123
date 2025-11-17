@echo off
cd /d "%~dp0"
echo Starting GitHub upload...
python upload_to_github.py
echo.
echo Press any key to continue...
pause > nul
