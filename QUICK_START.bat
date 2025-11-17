@echo off
echo ========================================
echo  QUICK START - FACE CLUSTERING APP
echo ========================================
echo.
echo This batch file will:
echo 1. Check current directory
echo 2. Start the server
echo 3. Open browser (if possible)
echo.
echo Press any key to continue...
pause > nul

cls
echo ========================================
echo  FACE CLUSTERING APPLICATION
echo ========================================
echo.
echo Current directory: %CD%
echo Target directory: %~dp0
echo.

cd /d "%~dp0"
echo Changed to: %CD%
echo.

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo.

echo Starting server...
echo Open browser to: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py

echo.
echo Server stopped.
echo Press any key to exit...
pause > nul
