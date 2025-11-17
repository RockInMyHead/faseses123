#!/usr/bin/env python3
"""Принудительное использование buffalo_l модели"""

import os
import sys
import gc
import psutil
from pathlib import Path

def check_memory():
    """Проверка доступной памяти"""
    print("Checking system memory...")

    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    used_gb = memory.used / (1024**3)

    print(f"Total RAM: {total_gb:.1f} GB")
    print(f"Available RAM: {available_gb:.1f} GB")
    print(f"Used RAM: {used_gb:.1f} GB")

    # Для buffalo_l нужно минимум 1GB свободной RAM
    if available_gb < 1.0:
        print("WARNING: Less than 1GB RAM available. buffalo_l may fail.")
        return False
    else:
        print("Memory looks OK for buffalo_l")
        return True

def clear_memory():
    """Очистка памяти"""
    print("Clearing memory...")
    gc.collect()

    # Попробовать очистить кэш insightface
    try:
        import insightface
        if hasattr(insightface, 'model_zoo'):
            insightface.model_zoo._model_cache.clear()
        print("InsightFace cache cleared")
    except:
        pass

def force_buffalo_l():
    """Принудительное использование buffalo_l"""
    print("FORCING BUFFALO_L MODEL")
    print("=" * 50)

    # Проверяем память
    memory_ok = check_memory()

    # Очищаем память
    clear_memory()

    # Проверяем импорты
    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("Imports OK")
    except ImportError as e:
        print(f"Import failed: {e}")
        print("Install: pip install insightface onnxruntime")
        return False

    # Проверяем/скачиваем модели buffalo_l
    print("\nEnsuring buffalo_l models are available...")

    # Принудительно инициализируем модель
    try:
        print("Trying buffalo_l with minimal settings...")
        app = FaceAnalysis(name="buffalo_l", root=None)
        print("FaceAnalysis created")

        # Пробуем с разными настройками
        settings_to_try = [
            {"ctx_id": 0, "det_size": (320, 320)},  # Минимальный размер
            {"ctx_id": 0, "det_size": (416, 416)},  # Средний размер
            {"ctx_id": 0, "det_size": (512, 512)},  # Большой размер
            {"ctx_id": 0, "det_size": (640, 640)},  # Оригинальный размер
        ]

        for i, settings in enumerate(settings_to_try):
            try:
                print(f"Trying settings {i+1}: {settings}")
                app.prepare(**settings)
                print(f"SUCCESS with settings: {settings}")
                return True
            except Exception as e:
                print(f"Settings {i+1} failed: {e}")
                continue

        print("All settings failed for buffalo_l")
        return False

    except Exception as e:
        print(f"Failed to create FaceAnalysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_main_for_buffalo_l():
    """Обновляем main.py для принудительного использования buffalo_l"""
    main_file = "main.py"

    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Меняем порядок моделей - buffalo_l первой
        if 'models_to_try = ["buffalo_s", "antelopev2", "buffalo_m", "buffalo_l"]' in content:
            new_content = content.replace(
                'models_to_try = ["buffalo_s", "antelopev2", "buffalo_m", "buffalo_l"]',
                'models_to_try = ["buffalo_l", "buffalo_s", "antelopev2", "buffalo_m"]'
            )

            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print("Updated main.py to prefer buffalo_l")
            return True
        else:
            print("Could not find models list in main.py")
            return False

    except Exception as e:
        print(f"Failed to update main.py: {e}")
        return False

def main():
    print("BUFFALO_L FORCE LOADER")
    print("=" * 50)

    # Сначала пробуем принудительно загрузить buffalo_l
    if force_buffalo_l():
        print("\nSUCCESS: buffalo_l is working!")
        print("Updating main.py to use buffalo_l...")

        if update_main_for_buffalo_l():
            print("DONE: System will now use buffalo_l")
            return True
        else:
            print("buffalo_l works but failed to update main.py")
            return True

    else:
        print("\nFAILED: Cannot make buffalo_l work on this system")
        print("\nRECOMMENDATIONS:")
        print("1. Restart your computer and try again")
        print("2. Close other memory-intensive applications")
        print("3. Increase virtual memory/page file size")
        print("4. Use a system with more RAM")
        print("5. Consider using buffalo_s as alternative (still very good)")

        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nNow run: python main.py")
        print("You should see: 'Используется InsightFace (buffalo_l) - лучшее качество распознавания'")
    else:
        print("\nConsider using buffalo_s instead:")
        print("python switch_to_light_model.py")
    sys.exit(0 if success else 1)
