@echo off
echo ========================================
echo  FACE CLUSTERING APPLICATION
echo ========================================
echo.
echo Starting the Face Clustering Server...
echo.
echo After startup, open your browser to:
echo http://localhost:8000
echo.
pause

cd /d "%~dp0"
python main.py

echo.
echo Server stopped.
echo Press any key to exit...
pause > nul
