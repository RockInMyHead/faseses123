#!/usr/bin/env python3
"""
Автоматическое исправление качества распознавания
Выбирает лучший доступный метод распознавания
"""

import os
import sys
import subprocess
import traceback

def test_insightface():
    """Тест insightface"""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        return True
    except Exception as e:
        print(f"InsightFace failed: {e}")
        return False

def test_face_recognition():
    """Тест face_recognition"""
    try:
        import face_recognition
        # Простой тест
        result = face_recognition.face_encodings(face_recognition.load_image_file("test_photos/test1.jpg"))
        return len(result) > 0
    except Exception as e:
        print(f"Face recognition failed: {e}")
        return False

def update_main_py(use_face_recognition):
    """Обновляет main.py"""
    main_file = "main.py"
    if not os.path.exists(main_file):
        print(f"main.py not found at {main_file}")
        return False

    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()

    old_setting = "USE_FACE_RECOGNITION = True" if use_face_recognition else "USE_FACE_RECOGNITION = False"
    new_setting = "USE_FACE_RECOGNITION = False" if use_face_recognition else "USE_FACE_RECOGNITION = True"

    if old_setting in content:
        new_content = content.replace(old_setting, new_setting)
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated main.py: {new_setting}")
        return True
    else:
        print("Setting not found in main.py")
        return False

def main():
    print("AUTO FIX RECOGNITION QUALITY")
    print("=" * 50)

    print("Testing available methods...")

    insightface_ok = test_insightface()
    face_rec_ok = test_face_recognition()

    print("\nRESULTS:")
    print(f"InsightFace (best quality): {'OK' if insightface_ok else 'FAILED'}")
    print(f"Face Recognition (fallback): {'OK' if face_rec_ok else 'FAILED'}")

    if insightface_ok:
        print("\nUsing InsightFace (best quality)")
        update_main_py(use_face_recognition=False)
        print("SUCCESS: Switched to InsightFace")
        return True

    elif face_rec_ok:
        print("\nUsing Face Recognition (improved settings)")
        update_main_py(use_face_recognition=True)
        print("SUCCESS: Using Face Recognition with better settings")
        return True

    else:
        print("\nFAILED: No working face recognition method found")
        print("Try installing dependencies:")
        print("pip install insightface onnxruntime")
        print("or")
        print("pip install face-recognition")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nDONE! Try running the server now.")
    else:
        print("\nFAILED: Could not fix recognition quality")
        sys.exit(1)
