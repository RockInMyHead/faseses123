@echo off
cd /d "%~dp0"
echo ========================================
echo  UPLOADING TO GITHUB
echo  Repository: https://github.com/RockInMyHead/face_0711.git
echo ========================================
echo.
python github_upload_final.py
echo.
echo ========================================
echo  UPLOAD COMPLETE!
echo  Check: https://github.com/RockInMyHead/face_0711.git
echo ========================================
echo.
echo Press any key to continue...
pause > nul
