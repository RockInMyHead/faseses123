#!/usr/bin/env python3
"""
Simple launcher for Face Clustering Application
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def main():
    print("FACE CLUSTERING APPLICATION LAUNCHER")
    print("=" * 50)

    # Определяем директорию скрипта
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}")
    print(f"Current directory: {os.getcwd()}")

    # Меняем на директорию скрипта
    os.chdir(script_dir)
    print(f"Changed to: {os.getcwd()}")

    # Проверяем наличие main.py
    main_py = script_dir / "main.py"
    if not main_py.exists():
        print(f"ERROR: main.py not found in {script_dir}")
        input("Press Enter to exit...")
        return

    print("\nStarting server...")
    print("Open your browser to: http://localhost:8000")
    print("Press Ctrl+C to stop")

    try:
        # Пробуем открыть браузер
        try:
            webbrowser.open("http://localhost:8000")
        except:
            pass  # Игнорируем ошибки браузера

        # Запускаем сервер
        subprocess.run([sys.executable, "main.py"], cwd=script_dir)

    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

    print("\nServer stopped.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
