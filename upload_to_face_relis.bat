@echo off
echo ========================================
echo  UPLOAD TO FACE_RELIS_PROJECT
echo ========================================
echo.
echo This will upload your project to:
echo https://github.com/RockInMyHead/face_relis_project.git
echo.
echo Repository already exists with 14 commits.
echo This will add new files and update existing ones.
echo.
pause

echo.
echo Starting upload process...
echo.

cd /d "%~dp0"
python github_upload_final.py

echo.
echo ========================================
echo  UPLOAD TO FACE_RELIS COMPLETE
echo ========================================
echo.
echo Check your repository:
echo https://github.com/RockInMyHead/face_relis_project.git
echo.
pause
