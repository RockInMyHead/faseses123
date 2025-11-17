@echo off
echo 🔧 Установка InsightFace для Windows...
echo 📁 Рабочая директория: %cd%
echo.

echo 🔍 Определяем версию Python...
python --version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Версия Python: %PYTHON_VERSION%
echo.

echo 📦 Проверяем, установлен ли уже InsightFace...
python -c "import insightface; print('InsightFace version:', insightface.__version__)" 2>nul
if not errorlevel 1 (
    echo ✅ InsightFace уже установлен!
    goto :test_insightface
)

echo ⬇️ Устанавливаем зависимости InsightFace...
echo.

REM Сначала устанавливаем зависимости
echo 📥 Шаг 1: Устанавливаем onnxruntime...
pip install onnxruntime
if errorlevel 1 (
    echo ⚠️ onnxruntime не установился
)

echo 📥 Шаг 2: Устанавливаем opencv-python (если не установлен)...
pip install opencv-python
if errorlevel 1 (
    echo ⚠️ opencv-python не установился
)

echo 📥 Шаг 3: Устанавливаем numpy (если не установлен)...
pip install numpy
if errorlevel 1 (
    echo ⚠️ numpy не установился
)

echo 📥 Шаг 4: Устанавливаем InsightFace...
echo 🔧 Попытка 1: Стандартная установка...
pip install insightface
if not errorlevel 1 goto :test_insightface

echo 🔧 Попытка 2: Установка для пользователя...
pip install --user insightface
if not errorlevel 1 goto :test_insightface

echo 🔧 Попытка 3: Установка без зависимостей...
pip install insightface --no-deps
if not errorlevel 1 goto :test_insightface

echo 🔧 Попытка 4: Установка из GitHub...
pip install git+https://github.com/deepinsight/insightface.git
if not errorlevel 1 goto :test_insightface

echo ❌ Все попытки установки InsightFace провалились
echo.
echo 🔧 РУЧНЫЕ СПОСОБЫ УСТАНОВКИ:
echo.
echo 📋 ВАРИАНТ 1 - Скачать wheel файл:
echo    1. Откройте https://pypi.org/project/insightface/#files
echo    2. Найдите wheel для вашей версии Python
echo    3. Скачайте файл
echo    4. pip install путь\к\скачанному\файлу.whl
echo.
echo 📋 ВАРИАНТ 2 - Использовать conda:
echo    conda install -c conda-forge insightface
echo.
echo 📋 ВАРИАНТ 3 - Установка в виртуальном окружении:
echo    python -m venv insightface_env
echo    insightface_env\Scripts\activate
echo    pip install insightface
echo.
echo 📋 ВАРИАНТ 4 - Проверьте версию Python (рекомендуется 3.8-3.11)
echo.
pause
exit /b 1

:test_insightface
echo.
echo 🔍 Тестируем работу InsightFace...
echo.

REM Тестируем базовый импорт
python -c "import insightface; print('✅ InsightFace импортируется')" 2>nul
if errorlevel 1 (
    echo ❌ InsightFace не импортируется
    goto :manual_install
)

REM Тестируем создание FaceAnalysis
python -c "import insightface; fa = insightface.app.FaceAnalysis(); print('✅ FaceAnalysis создается')" 2>nul
if errorlevel 1 (
    echo ❌ FaceAnalysis не создается
    echo 🔧 Возможно, нужны дополнительные зависимости
    goto :manual_install
)

REM Тестируем загрузку модели
python -c "import insightface; fa = insightface.app.FaceAnalysis(); fa.prepare(ctx_id=-1); print('✅ Модель загружается')" 2>nul
if errorlevel 1 (
    echo ❌ Модель не загружается
    echo 🔧 Возможно, проблема с загрузкой модели
    goto :manual_install
)

echo ✅ InsightFace полностью работает!
echo 🎯 Теперь FaceSort должен работать корректно
goto :end

:manual_install
echo.
echo ❌ InsightFace установлен, но не работает корректно
echo.
echo 🔧 ДОПОЛНИТЕЛЬНЫЕ ШАГИ:
echo.
echo 1. Убедитесь, что установлены:
echo    pip install onnxruntime opencv-python numpy
echo.
echo 2. Попробуйте переустановить:
echo    pip uninstall insightface
echo    pip install insightface --no-cache-dir
echo.
echo 3. Проверьте, что нет конфликтов версий
echo.
echo 4. Попробуйте старую версию:
echo    pip install insightface==0.7.3
echo.

:end
echo.
echo 🎉 Установка завершена!
echo 📋 Теперь запустите start_server_windows.bat
pause
