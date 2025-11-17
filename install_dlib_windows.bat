@echo off
echo 🔧 Установка dlib для Windows...
echo 📁 Рабочая директория: %cd%
echo.

echo 🔍 Определяем версию Python...
python --version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Версия Python: %PYTHON_VERSION%
echo.

echo 📦 Проверяем, установлен ли уже dlib...
python -c "import dlib; print('dlib version:', dlib.__version__)" 2>nul
if not errorlevel 1 (
    echo ✅ dlib уже установлен!
    goto :face_recognition
)

echo ⬇️ Пробуем установить dlib...
echo 🔧 dlib - сложная библиотека, может потребоваться время...
echo.

REM Сначала пробуем pip install dlib (может автоматически найти wheel)
echo 📦 Попытка 1: Стандартная установка через pip...
pip install dlib
if not errorlevel 1 goto :check_install

REM Если не получилось, пробуем с --only-binary
echo 📦 Попытка 2: Только бинарные файлы...
pip install dlib --only-binary all
if not errorlevel 1 goto :check_install

REM Если все еще не получилось, предлагаем альтернативы
echo ❌ Автоматическая установка dlib не удалась
echo.
echo 🔧 Ручные варианты установки dlib:
echo.
echo 📋 ВАРИАНТ 1 - Скачать wheel файл вручную:
echo    1. Откройте https://pypi.org/project/dlib/#files
echo    2. Найдите файл вида: dlib-19.24.2-cp311-cp311-win_amd64.whl
echo       ^(cp311 для Python 3.11, cp310 для Python 3.10 и т.д.^)
echo    3. Скачайте файл
echo    4. pip install путь\к\скачанному\файлу.whl
echo.
echo 📋 ВАРИАНТ 2 - Использовать conda:
echo    conda install -c conda-forge dlib
echo.
echo 📋 ВАРИАНТ 3 - Установка через Chocolatey ^(если установлен^):
echo    choco install dlib
echo.
echo 📋 ВАРИАНТ 4 - Сборка из исходников ^(самый сложный^):
echo    pip install cmake
echo    pip install dlib --no-binary dlib
echo    ^(Требует Visual Studio Build Tools^)
echo.
echo ❓ Выберите вариант и нажмите Enter для продолжения без dlib...
pause
goto :face_recognition

:check_install
echo.
echo 🔍 Проверяем установку dlib...
python -c "import dlib; print('✅ dlib успешно установлен! Версия:', dlib.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ dlib не установлен корректно
    goto :face_recognition
) else (
    echo ✅ dlib успешно установлен!
)

:face_recognition
echo.
echo 🎯 Устанавливаем face-recognition...
pip install face-recognition face-recognition-models
if errorlevel 1 (
    echo ⚠️ face-recognition не установился
    echo 🔧 Возможно, потребуется установить dlib сначала
) else (
    echo ✅ face-recognition установлен!
)

echo.
echo 🎉 Установка завершена!
echo 📋 Теперь можно запускать start_server_windows.bat
pause
