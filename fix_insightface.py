#!/usr/bin/env python3
"""Исправление проблем с insightface и ONNX моделями"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Выполняет команду и выводит результат"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print(f"✅ {description} - успешно")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} - ошибка:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {description} - исключение: {e}")
        return False

def fix_insightface():
    """Комплексное исправление проблем с insightface"""

    print("🚀 Начинаем исправление проблем с insightface...")

    # Шаг 1: Очистка кэша
    print("\n1. Очистка кэша моделей...")
    cache_script = Path(__file__).parent / "clear_insightface_cache.py"
    if cache_script.exists():
        run_command(f"python {cache_script}", "Очистка кэша insightface")
    else:
        print("⚠️ Скрипт очистки кэша не найден")

    # Шаг 2: Удаление проблемных пакетов
    print("\n2. Удаление insightface и onnxruntime...")
    run_command("pip uninstall -y insightface onnxruntime onnxruntime-gpu", "Удаление пакетов")

    # Шаг 3: Установка совместимых версий
    print("\n3. Установка совместимых версий...")

    # Сначала попробуем onnxruntime CPU
    success = run_command("pip install onnxruntime==1.15.1", "Установка onnxruntime 1.15.1")

    if success:
        # Затем insightface совместимой версии
        success = run_command("pip install insightface==0.7.3", "Установка insightface 0.7.3")

    if not success:
        print("❌ Проблема с установкой. Попробуем альтернативный подход...")
        # Fallback: попробуем другие версии
        run_command("pip install onnxruntime==1.14.1", "Установка onnxruntime 1.14.1")
        run_command("pip install insightface==0.7.1", "Установка insightface 0.7.1")

    # Шаг 4: Проверка установки
    print("\n4. Проверка установки...")
    test_script = Path(__file__).parent / "test_insightface.py"
    if test_script.exists():
        result = run_command(f"python {test_script}", "Тестирование insightface")
        if result:
            print("🎉 Исправление завершено успешно!")
            return True
        else:
            print("❌ Тест не прошел. Попробуем другие решения...")
    else:
        print("⚠️ Тестовый скрипт не найден")

    # Шаг 5: Альтернативное решение - переключение на face_recognition
    print("\n5. Альтернативное решение - переключение на face_recognition...")
    print("💡 Рекомендуется использовать face_recognition вместо insightface для большей стабильности")
    print("   Измените main.py для использования face_recognition или dlib")

    return False

if __name__ == "__main__":
    success = fix_insightface()
    if not success:
        print("\n🔧 Ручные действия для исправления:")
        print("1. Очистите кэш: python clear_insightface_cache.py")
        print("2. Переустановите: pip uninstall insightface onnxruntime && pip install insightface==0.7.3 onnxruntime==1.15.1")
        print("3. Или переключитесь на face_recognition: раскомментируйте соответствующие строки в requirements.txt")
        sys.exit(1)
