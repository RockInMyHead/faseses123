@echo off
echo ========================================
echo  INSTALLING DEPENDENCIES
echo ========================================
echo.
echo This will install all required packages for
echo the Face Clustering Application.
echo.
echo Required packages:
echo - FastAPI (web server)
echo - InsightFace (AI face recognition)
echo - OpenCV, Pillow (image processing)
echo - Faiss (clustering)
echo - And more...
echo.
pause

cd /d "%~dp0"

echo.
echo Installing dependencies from requirements-win.txt...
echo.

pip install -r requirements-win.txt

echo.
echo ========================================
echo  INSTALLATION COMPLETE
echo ========================================
echo.
echo If there were no errors above, you can now run:
echo RUN_PROJECT.bat
echo.
echo If you got errors, try:
echo pip install --upgrade pip
echo then run this script again.
echo.
pause
