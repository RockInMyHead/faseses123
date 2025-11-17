@echo off
cd /d "%~dp0"
echo Testing buffalo_l...
python test_buffalo_l_final.py
echo.
echo Press any key to continue...
pause > nul
