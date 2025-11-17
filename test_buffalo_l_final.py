#!/usr/bin/env python3
"""Финальный тест buffalo_l"""

import os
import sys

def main():
    print("BUFFALO_L FINAL TEST")
    print("=" * 30)

    try:
        # Проверяем, можем ли мы импортировать insightface
        import insightface
        from insightface.app import FaceAnalysis

        print("Testing buffalo_l initialization...")

        # Пробуем инициализировать с минимальными настройками
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(320, 320))  # Минимальный размер

        print("SUCCESS: buffalo_l is working!")
        print("The system will now use buffalo_l for best accuracy.")

        # Обновляем main.py для использования buffalo_l
        main_file = "main.py"
        if os.path.exists(main_file):
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Меняем порядок моделей
            if 'models_to_try = ["buffalo_s", "antelopev2", "buffalo_m", "buffalo_l"]' in content:
                new_content = content.replace(
                    'models_to_try = ["buffalo_s", "antelopev2", "buffalo_m", "buffalo_l"]',
                    'models_to_try = ["buffalo_l", "buffalo_s", "antelopev2", "buffalo_m"]'
                )
                with open(main_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print("Updated main.py to prefer buffalo_l")

        return True

    except Exception as e:
        print(f"FAILED: {e}")
        print("\nRecommendation: Use buffalo_s instead")
        print("Run: python switch_to_light_model.py")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nNow run: python main.py")
        print("You should see: 'Используется InsightFace (buffalo_l) - лучшее качество распознавания'")
    sys.exit(0 if success else 1)
