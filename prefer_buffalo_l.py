#!/usr/bin/env python3
"""Изменение приоритета моделей - buffalo_l первой"""

import os

def main():
    print("Setting buffalo_l as preferred model")
    print("=" * 50)

    main_file = "main.py"

    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Ищем строку с models_to_try
        old_line = 'models_to_try = ["buffalo_s", "antelopev2", "buffalo_m", "buffalo_l"]'
        new_line = 'models_to_try = ["buffalo_l", "buffalo_s", "antelopev2", "buffalo_m"]'

        if old_line in content:
            new_content = content.replace(old_line, new_line)

            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print("SUCCESS: buffalo_l is now the preferred model")
            print("System will try buffalo_l first, then fall back to others")
            return True
        else:
            print("Could not find models list in main.py")
            return False

    except Exception as e:
        print(f"Error updating main.py: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nNow run: python main.py")
        print("You should see: 'Используется InsightFace (buffalo_l) - лучшее качество распознавания'")
    else:
        print("\nFailed to update preferences")
    exit(0 if success else 1)
