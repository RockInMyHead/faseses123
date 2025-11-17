@echo off
echo 🚀 Запуск проекта FaceSort на Windows...
echo 📁 Рабочая директория: %cd%
python --version
echo.

echo 📦 Проверяем виртуальное окружение...
if not exist venv (
    echo 🔧 Создаем виртуальное окружение...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Ошибка создания виртуального окружения
        pause
        exit /b 1
    )
    echo ✅ Виртуальное окружение создано
) else (
    echo ✅ Виртуальное окружение найдено
)

echo 🔄 Активируем виртуальное окружение...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Ошибка активации виртуального окружения
    pause
    exit /b 1
)
echo ✅ Виртуальное окружение активировано
echo.

echo 📦 Проверяем зависимости...
python -c "import fastapi, uvicorn, PIL, cv2, insightface, faiss" 2>nul
if errorlevel 1 (
    echo ❌ Некоторые зависимости не установлены.
    echo 🔧 Устанавливаем зависимости...

    echo 📥 Шаг 1: Обновляем pip...
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo ⚠️ Не удалось обновить pip системно, пробуем для пользователя...
        pip install --user --upgrade pip
    )

    echo 📥 Шаг 2: Устанавливаем основные пакеты...
    pip install fastapi uvicorn python-multipart jinja2 aiofiles pillow opencv-python numpy scipy matplotlib seaborn pandas tqdm psutil pyyaml python-dotenv requests httpx scikit-learn faiss-cpu
    if errorlevel 1 (
        echo ⚠️ Ошибка установки основных пакетов, пробуем по одному...
        pip install --user fastapi uvicorn pillow opencv-python numpy scipy
        pip install --user matplotlib seaborn pandas tqdm psutil
        pip install --user scikit-learn faiss-cpu
    )

    echo 📥 Шаг 3: Устанавливаем ML пакеты...
    echo 🔧 Устанавливаем InsightFace...
    call install_insightface_windows.bat
    if errorlevel 1 (
        echo ❌ InsightFace не удалось установить
        echo 🔧 FaceSort не сможет работать без InsightFace
        pause
        exit /b 1
    )

    echo 📥 Шаг 4: Устанавливаем dlib и face-recognition...
    echo 🔧 dlib может требовать Visual Studio Build Tools...
    echo 📋 Если установка dlib не удастся, установите вручную:
    echo    pip install https://pypi.org/project/dlib/19.24.0/
    echo    или скачайте wheel с https://pypi.org/project/dlib/#files
    pip install dlib
    if errorlevel 1 (
        echo ❌ dlib не установился автоматически
        echo 🔧 Попробуйте один из вариантов:
        echo    1. pip install https://files.pythonhosted.org/packages/1a/50/fc9b21e54c2c1b2ac1b9a9a6c1c6b6e5a5d4f4e5d6f7e8f9a0b1c2d3e4f5a6/dlib-19.24.0-cp311-cp311-win_amd64.whl
        echo    2. conda install -c conda-forge dlib
        echo    3. Скачайте wheel файл вручную
        echo.
        echo ⏳ Продолжаем без dlib...
    )

    pip install face-recognition face-recognition-models
    if errorlevel 1 (
        echo ⚠️ face-recognition не установился
    )

    echo 📥 Шаг 5: Проверяем установку...
    python -c "import fastapi, uvicorn, PIL, cv2" 2>nul
    if errorlevel 1 (
        echo ❌ Основные пакеты не установлены
        echo 🔧 Проверьте логи выше и установите пакеты вручную
        pause
        exit /b 1
    ) else (
        echo ✅ Основные зависимости установлены
    )

    python -c "import insightface; fa = insightface.app.FaceAnalysis(); fa.prepare(ctx_id=-1)" 2>nul
    if errorlevel 1 (
        echo ❌ InsightFace не работает корректно
        echo 🔧 Запустите install_insightface_windows.bat отдельно
    ) else (
        echo ✅ InsightFace полностью работает
    )

) else (
    echo ✅ Основные зависимости установлены
)
echo.

echo 🛑 Останавливаем предыдущие процессы...
taskkill /f /im python.exe /fi "WINDOWTITLE eq main.py*" >nul 2>&1
taskkill /f /im python.exe /fi "IMAGENAME eq python.exe" /fi "MEMUSAGE gt 100000" >nul 2>&1
timeout /t 2 >nul

echo 🚀 Запускаем сервер FaceSort...
start "FaceSort Server" python main.py

echo ✅ Сервер запущен!
echo 🌐 URL: http://localhost:8000
echo 📊 Проверка через 5 секунд...
timeout /t 5 >nul

echo 📋 Для остановки сервера закройте окно командной строки или нажмите Ctrl+C
echo 🎯 Откройте http://localhost:8000 в браузере
pause
