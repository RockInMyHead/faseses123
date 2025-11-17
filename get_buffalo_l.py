#!/usr/bin/env python3
"""
ПОЛУЧЕНИЕ BUFFALO_L МОДЕЛИ
Комплексное решение для принудительного использования buffalo_l
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Выполняет команду"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"[OK] {description}")
            return True
        else:
            print(f"[FAILED] {description}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"[EXCEPTION] {description}: {e}")
        return False

def main():
    print("BUFFALO_L ACQUISITION SYSTEM")
    print("=" * 50)
    print("Цель: Заставить buffalo_l работать на вашей системе")
    print("=" * 50)

    steps = [
        ("Очистка кэша моделей", "python clear_insightface_cache.py"),
        ("Принудительная загрузка buffalo_l", "python force_buffalo_l.py"),
        ("Установка приоритета buffalo_l", "python prefer_buffalo_l.py"),
        ("Финальная диагностика", "python diagnose_insightface.py --force-buffalo-l"),
    ]

    for description, command in steps:
        print(f"\n{description}:")
        print("-" * 40)
        success = run_command(command, description)
        if not success:
            print(f"[FAILED] Step '{description}' failed")
            break
    else:
        print("\n" + "=" * 50)
        print("[SUCCESS] ALL STEPS COMPLETED!")
        print("Теперь buffalo_l должен работать")
        print("\nЗапустите сервер:")
        print("python main.py")
        print("\nОжидайте сообщение:")
        print("'Используется InsightFace (buffalo_l) - лучшее качество распознавания'")
        return True

    print("\n" + "=" * 50)
    print("[FAILED] COULD NOT MAKE BUFFALO_L WORK")
    print("\nАЛЬТЕРНАТИВНЫЕ РЕШЕНИЯ:")
    print("1. Перезагрузите компьютер и попробуйте снова")
    print("2. Закройте все другие программы")
    print("3. Увеличьте виртуальную память Windows:")
    print("   - Правой кнопкой на 'Этот компьютер' > Свойства")
    print("   - Дополнительные параметры > Быстродействие > Параметры")
    print("   - Дополнительно > Виртуальная память > Изменить")
    print("   - Снимите 'Автоматически' и установите минимум 4096MB")
    print("4. Используйте buffalo_s - она почти такая же точная:")
    print("   python switch_to_light_model.py")
    print("5. Попробуйте на другом компьютере с больше RAM")

    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
