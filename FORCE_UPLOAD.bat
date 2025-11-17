@echo off
echo ========================================
echo  FORCE UPLOAD TO FACE_RELIS_PROJECT
echo ========================================
echo.
echo This will FORCE upload your project to:
echo https://github.com/RockInMyHead/face_relis_project.git
echo.
echo WARNING: This uses --force push!
echo Existing commits may be overwritten.
echo.
pause

echo.
echo Starting FORCE upload process...
echo.

cd /d "%~dp0"
python force_upload_to_github.py

echo.
echo ========================================
echo  FORCE UPLOAD COMPLETE
echo ========================================
echo.
echo Check your repository:
echo https://github.com/RockInMyHead/face_relis_project.git
echo.
pause
