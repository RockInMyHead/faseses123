#!/usr/bin/env python3
"""
Проверка работы сервера FaceSort
"""

import requests
import time
import sys

def check_server():
    """Проверяет работу сервера"""
    print("🔍 Проверяем работу сервера FaceSort...")
    
    try:
        # Проверяем доступность сервера
        response = requests.get("http://localhost:8001/api/tasks", timeout=5)
        if response.status_code == 200:
            print("✅ Сервер работает на http://localhost:8001")
            print(f"📊 Ответ API: {response.json()}")
            return True
        else:
            print(f"❌ Сервер отвечает с кодом: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Сервер недоступен - возможно не запущен")
        return False
    except requests.exceptions.Timeout:
        print("❌ Таймаут подключения к серверу")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_common_photos_api():
    """Тестирует API общих фотографий"""
    print("\n🧪 Тестируем API общих фотографий...")
    
    try:
        test_data = {
            "rootPath": "/Users/artembutko/Desktop/116_Даша-2",
            "commonFolders": ["/Users/artembutko/Desktop/116_Даша-2/Младшая/общие"]
        }
        
        response = requests.post(
            "http://localhost:8001/api/process-common-photos",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API общих фотографий работает")
            print(f"📊 Результат: {result}")
            return True
        else:
            print(f"❌ API ошибка: {response.status_code}")
            print(f"📊 Ответ: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка тестирования API: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Проверка исправленного проекта FaceSort")
    print("=" * 50)
    
    # Проверяем сервер
    if check_server():
        print("\n🎯 Сервер работает! Тестируем API...")
        test_common_photos_api()
    else:
        print("\n❌ Сервер не работает. Запустите: python main.py")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Проверка завершена!")
    print("🌐 Откройте http://localhost:8000 в браузере")
    print("🔧 Используйте кнопку '📂 Общие' для тестирования")
