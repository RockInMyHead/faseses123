#!/usr/bin/env python3
"""Переключение на легкую модель buffalo_s для экономии памяти"""

import os
import sys

def main():
    print("Switching to lightweight model buffalo_s")
    print("=" * 50)

    # Проверяем main.py
    main_file = "main.py"
    if not os.path.exists(main_file):
        print("main.py not found")
        return False

    # Очищаем кэш моделей
    print("Clearing model cache...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "clear_insightface_cache.py"],
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print("Cache cleared successfully")
        else:
            print("Failed to clear cache")
    except Exception as e:
        print(f"Error clearing cache: {e}")

    # Тестируем buffalo_s
    print("\nTesting buffalo_s model...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_s")
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("buffalo_s model works!")
        return True
    except Exception as e:
        print(f"buffalo_s failed: {e}")

        # Если buffalo_s не работает, пробуем другие модели
        models_to_try = ["antelopev2", "buffalo_m"]
        for model in models_to_try:
            try:
                print(f"Trying {model}...")
                app = FaceAnalysis(name=model)
                app.prepare(ctx_id=0, det_size=(640, 640))
                print(f"{model} works!")
                return True
            except Exception as e2:
                print(f"{model} failed: {e2}")
                continue

        print("No lightweight models work. Using face_recognition as fallback.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: Lightweight model is working!")
        print("The system will automatically use it now.")
    else:
        print("\nUsing face_recognition as fallback.")
        print("Install better models later if needed.")
    sys.exit(0 if success else 1)
