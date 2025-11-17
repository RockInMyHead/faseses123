#!/usr/bin/env python3
"""Диагностика проблем с insightface"""

import os
import sys
import traceback
from pathlib import Path

def check_imports():
    """Проверка импортов"""
    print("Checking imports...")

    try:
        import insightface
        print(f"insightface: OK (version {getattr(insightface, '__version__', 'unknown')})")
        return True
    except ImportError as e:
        print(f"insightface: FAILED - {e}")
        return False

    try:
        import onnxruntime
        print(f"onnxruntime: OK (version {getattr(onnxruntime, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"onnxruntime: FAILED - {e}")

def check_models():
    """Проверка моделей"""
    print("\nChecking models...")

    # Пути к моделям
    home = Path.home()
    model_paths = [
        home / ".insightface" / "models" / "buffalo_l",
        home / "AppData" / "Local" / "insightface" / "models" / "buffalo_l",
        home / "AppData" / "Roaming" / "insightface" / "models" / "buffalo_l",
    ]

    for path in model_paths:
        if path.exists():
            print(f"Models found at: {path}")
            onnx_files = list(path.glob("*.onnx"))
            print(f"ONNX files: {len(onnx_files)}")
            for f in onnx_files:
                size = f.stat().st_size / (1024*1024)  # MB
                print(f"  - {f.name}: {size:.1f} MB")
            return True

    print("No buffalo_l models found")
    return False

def test_buffalo_l_force():
    """Специальный тест buffalo_l с оптимизациями"""
    print("\nForce testing buffalo_l...")

    try:
        import gc
        gc.collect()  # Очистка памяти

        from insightface.app import FaceAnalysis

        # Пробуем с минимальными настройками
        print("Trying buffalo_l with minimal memory usage...")

        # Сначала пробуем с маленьким det_size
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(320, 320))  # Минимальный размер для экономии памяти

        print("SUCCESS: buffalo_l loaded with reduced settings!")
        return "buffalo_l"

    except Exception as e:
        print(f"Force test failed: {e}")

        # Если не получилось, пробуем альтернативные подходы
        try:
            print("Trying alternative initialization...")

            # Очищаем кэш
            import insightface
            if hasattr(insightface, 'model_zoo'):
                insightface.model_zoo._model_cache.clear()

            # Пробуем снова
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", root=None)  # Без кэширования
            app.prepare(ctx_id=0, det_size=(416, 416))

            print("SUCCESS: buffalo_l loaded with alternative method!")
            return "buffalo_l"

        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return False

def test_initialization():
    """Тест инициализации"""
    print("\nTesting initialization...")

    # Сначала проверяем, хочет ли пользователь buffalo_l
    if "--force-buffalo-l" in sys.argv:
        result = test_buffalo_l_force()
        if result:
            return result

    # Обычный тест всех моделей
    models_to_try = ["buffalo_s", "antelopev2", "buffalo_m", "buffalo_l"]

    for model_name in models_to_try:
        try:
            print(f"Trying model: {model_name}")
            from insightface.app import FaceAnalysis
            print("FaceAnalysis import: OK")

            app = FaceAnalysis(name=model_name)
            print(f"FaceAnalysis init ({model_name}): OK")

            app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"Model prepare ({model_name}): OK")

            return model_name

        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue

    print("All models failed!")
    return False

def main():
    print("INSIGHTFACE DIAGNOSTICS")
    print("=" * 50)

    import_ok = check_imports()
    models_ok = check_models()
    working_model = test_initialization()

    print("\n" + "=" * 50)
    print("DIAGNOSTIC RESULTS:")
    print(f"Imports: {'OK' if import_ok else 'FAILED'}")
    print(f"Models: {'OK' if models_ok else 'FAILED'}")
    print(f"Working model: {working_model if working_model else 'NONE'}")

    if import_ok and working_model:
        print(f"\nRESULT: InsightFace is working with model '{working_model}'!")
        if working_model != "buffalo_l":
            print(f"NOTE: Using lightweight model '{working_model}' instead of heavy 'buffalo_l'")
            print("This saves memory but may be slightly less accurate.")
        return working_model
    else:
        print("\nRESULT: InsightFace has problems.")
        print("\nRECOMMENDATIONS:")
        if not import_ok:
            print("- Install insightface: pip install insightface onnxruntime")
        if not models_ok:
            print("- Clear model cache: python clear_insightface_cache.py")
            print("- Models will be downloaded automatically on first use")
        if not working_model:
            print("- Try lightweight models: buffalo_s, antelopev2")
            print("- For buffalo_l specifically: python force_buffalo_l.py")
            print("- Or force test: python diagnose_insightface.py --force-buffalo-l")
            print("- Or use face_recognition: pip install face-recognition")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
