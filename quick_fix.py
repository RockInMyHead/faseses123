#!/usr/bin/env python3
"""
Быстрое исправление проблем с распознаванием лиц.
Запустите этот скрипт для автоматического исправления.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_cmd(cmd, description):
    """Выполняет команду и выводит результат"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✅ {description} - успешно")
            return True
        else:
            print(f"❌ {description} - ошибка:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} - исключение: {e}")
        return False

def main():
    print("🚀 Быстрое исправление проблем с распознаванием лиц")
    print("=" * 60)

    # Шаг 1: Установка стабильных зависимостей
    print("\n📦 Шаг 1: Установка стабильных зависимостей (face_recognition)...")

    if run_cmd("pip install face-recognition==1.3.0 face-recognition-models==0.3.0", "Установка face_recognition"):
        print("✅ Face recognition установлен успешно!")
        print("💡 Теперь приложение будет использовать стабильную версию face_recognition вместо insightface")
        return True

    # Если face_recognition не удалось установить, пробуем исправить insightface
    print("\n📦 Шаг 2: Альтернативное исправление insightface...")

    # Очистка кэша
    cache_script = Path(__file__).parent / "clear_insightface_cache.py"
    if cache_script.exists():
        run_cmd(f"python {cache_script}", "Очистка кэша insightface")

    # Переустановка insightface
    run_cmd("pip uninstall -y insightface onnxruntime onnxruntime-gpu", "Удаление проблемных пакетов")
    run_cmd("pip install onnxruntime==1.15.1", "Установка onnxruntime 1.15.1")
    run_cmd("pip install insightface==0.7.3", "Установка insightface 0.7.3")

    # Тестирование
    test_script = Path(__file__).parent / "test_insightface.py"
    if test_script.exists():
        if run_cmd(f"python {test_script}", "Тестирование insightface"):
            print("✅ InsightFace исправлен успешно!")
            return True

    print("❌ Не удалось автоматически исправить проблему")
    print("\n🔧 Ручные действия:")
    print("1. Попробуйте перезагрузить компьютер")
    print("2. Проверьте подключение к интернету")
    print("3. Установите зависимости вручную:")
    print("   pip install face-recognition==1.3.0 face-recognition-models==0.3.0")
    print("4. Или для insightface:")
    print("   pip uninstall insightface onnxruntime")
    print("   pip install onnxruntime==1.15.1 insightface==0.7.3")
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Исправление завершено! Теперь попробуйте запустить распознавание.")
    else:
        print("\n❌ Автоматическое исправление не удалось.")
        sys.exit(1)
