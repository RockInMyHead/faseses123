#!/usr/bin/env python3
"""Скрипт для очистки кэша моделей insightface"""

import os
import shutil
from pathlib import Path

def clear_insightface_cache():
    """Очищает кэш моделей insightface"""
    home = Path.home()

    # Стандартные пути кэша insightface
    cache_paths = [
        home / ".insightface",  # Основной кэш
        home / ".cache" / "insightface",  # Альтернативный кэш
        home / "AppData" / "Local" / "insightface",  # Windows specific
        home / "AppData" / "Roaming" / "insightface",  # Windows specific
    ]

    print("Cleaning insightface models cache...")

    for cache_path in cache_paths:
        if cache_path.exists():
            try:
                if cache_path.is_file():
                    cache_path.unlink()
                    print(f"Deleted file: {cache_path}")
                else:
                    shutil.rmtree(cache_path)
                    print(f"Deleted folder: {cache_path}")
            except Exception as e:
                print(f"Warning: Failed to delete {cache_path}: {e}")
        else:
            print(f"Path does not exist: {cache_path}")

    print("Cache cleanup completed!")

if __name__ == "__main__":
    clear_insightface_cache()
