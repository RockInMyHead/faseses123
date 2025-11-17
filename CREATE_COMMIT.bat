@echo off
echo ========================================
echo  CREATE COMMIT FOR GITHUB UPLOAD
echo ========================================
echo.
echo This will create a dummy commit and upload to:
echo https://github.com/RockInMyHead/face_relis_project.git
echo.

cd /d "%~dp0"
python create_dummy_commit.py

echo.
echo ========================================
echo  COMMIT CREATED
echo ========================================
echo.
echo Check your repository:
echo https://github.com/RockInMyHead/face_relis_project.git
echo.
pause
