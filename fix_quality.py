#!/usr/bin/env python3
"""
Исправление качества распознавания - переключение на insightface
"""

import os
import sys

def main():
    print("Fixing recognition quality")
    print("=" * 50)
    print("Problem: Using face_recognition instead of insightface")
    print("Solution: Switching to insightface (ArcFace)")

    # Проверяем main.py
    main_file = "main.py"
    if os.path.exists(main_file):
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if "USE_FACE_RECOGNITION = True" in content:
            print("Found USE_FACE_RECOGNITION = True, fixing...")

            new_content = content.replace(
                "USE_FACE_RECOGNITION = True",
                "USE_FACE_RECOGNITION = False"
            )

            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print("Switched to insightface!")
        else:
            print("Already using insightface or setting not found")
    else:
        print("main.py file not found")
        return False

    print("\nTesting insightface...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "test_insightface_simple.py"],
                              capture_output=True, text=True, cwd=os.getcwd())

        if result.returncode == 0:
            print("InsightFace works!")
            print("Recognition quality should be much better now")
            return True
        else:
            print("Problem with insightface:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"Testing error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nDONE! Now start the server and try recognition.")
        print("Expect much better face clustering quality!")
    else:
        print("\nAutomatic fix failed.")
        print("Try manually:")
        print("1. Change USE_FACE_RECOGNITION = False in main.py")
        print("2. Run python test_insightface_simple.py to check")
        print("3. If insightface doesn't work, run python fix_insightface.py")
        sys.exit(1)
