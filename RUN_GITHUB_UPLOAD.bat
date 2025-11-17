@echo off
echo ========================================
echo  FACE CLUSTERING APP - GITHUB UPLOAD
echo ========================================
echo.
echo This will upload your project to:
echo https://github.com/RockInMyHead/face_relis_project.git
echo.
echo Make sure you have:
echo 1. Git installed
echo 2. GitHub account configured
echo 3. Push access to the repository
echo.
pause

echo.
echo Starting upload process...
echo.

cd /d "%~dp0"
python github_upload_final.py

echo.
echo ========================================
echo  UPLOAD PROCESS COMPLETE
echo ========================================
echo.
echo Check your repository:
echo https://github.com/RockInMyHead/face_relis_project.git
echo.
echo If upload failed, you may need to:
echo 1. Configure Git with your GitHub credentials
echo 2. Create a Personal Access Token
echo 3. Set up SSH keys
echo.
pause
