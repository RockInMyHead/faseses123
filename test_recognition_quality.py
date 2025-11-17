#!/usr/bin/env python3
"""Тест качества распознавания - сравнение face_recognition vs insightface"""

import os
import sys
from pathlib import Path
import time

def test_face_recognition():
    """Тест face_recognition"""
    print("🧪 Тестирование face_recognition...")
    try:
        import face_recognition
        print("✅ face_recognition импортирован")

        # Используем тестовые фото
        test_dir = Path("test_photos")
        if not test_dir.exists():
            print("❌ Папка test_photos не найдена")
            return False

        images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if not images:
            print("❌ Тестовые фото не найдены")
            return False

        print(f"📸 Найдено тестовых изображений: {len(images)}")

        start_time = time.time()
        total_faces = 0

        for img_path in images[:3]:  # Тестируем только первые 3
            try:
                image = face_recognition.load_image_file(str(img_path))
                face_locations = face_recognition.face_locations(image, model="hog")
                face_encodings = face_recognition.face_encodings(image, face_locations)

                print(f"  🖼 {img_path.name}: найдено {len(face_locations)} лиц")
                total_faces += len(face_locations)

            except Exception as e:
                print(f"  ❌ Ошибка с {img_path.name}: {e}")

        elapsed = time.time() - start_time
        print(".2f"        return True

    except ImportError as e:
        print(f"❌ face_recognition не установлен: {e}")
        return False

def test_insightface():
    """Тест insightface"""
    print("\n🧪 Тестирование insightface...")
    try:
        from insightface.app import FaceAnalysis
        print("✅ insightface импортирован")

        # Используем тестовые фото
        test_dir = Path("test_photos")
        if not test_dir.exists():
            print("❌ Папка test_photos не найдена")
            return False

        images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if not images:
            print("❌ Тестовые фото не найдены")
            return False

        print(f"📸 Найдено тестовых изображений: {len(images)}")

        # Инициализация модели
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))

        start_time = time.time()
        total_faces = 0

        for img_path in images[:3]:  # Тестируем только первые 3
            try:
                from PIL import Image
                import numpy as np

                # Загружаем изображение
                image = Image.open(img_path)
                image = np.array(image)

                faces = app.get(image)
                print(f"  🖼 {img_path.name}: найдено {len(faces)} лиц")
                total_faces += len(faces)

            except Exception as e:
                print(f"  ❌ Ошибка с {img_path.name}: {e}")

        elapsed = time.time() - start_time
        print(".2f"        return True

    except Exception as e:
        print(f"❌ Ошибка с insightface: {e}")
        return False

def compare_quality():
    """Сравнение качества распознавания"""
    print("🔍 СРАВНЕНИЕ КАЧЕСТВА РАСПОЗНАВАНИЯ")
    print("=" * 50)

    face_rec_ok = test_face_recognition()
    insightface_ok = test_insightface()

    print("\n📊 РЕЗУЛЬТАТЫ:")
    print("=" * 50)

    if face_rec_ok and insightface_ok:
        print("✅ Оба метода работают")
        print("💡 Рекомендация: Используйте insightface для лучшего качества")
        print("   Измените USE_FACE_RECOGNITION = False в main.py")

    elif insightface_ok:
        print("✅ InsightFace работает хорошо")
        print("💡 Рекомендация: Переключитесь на insightface")
        print("   Измените USE_FACE_RECOGNITION = False в main.py")

    elif face_rec_ok:
        print("⚠️ Только face_recognition работает")
        print("💡 Причина низкого качества:")
        print("   - face_recognition менее точный, чем insightface")
        print("   - Использует простую модель HOG вместо нейронных сетей")
        print("   - Хуже работает с углами, освещением, качеством фото")

    else:
        print("❌ Ни один метод не работает")
        print("🔧 Нужно установить зависимости")

if __name__ == "__main__":
    compare_quality()
