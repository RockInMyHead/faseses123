#!/usr/bin/env python3
import requests
import time

print("🔍 Проверяем сервер...")
time.sleep(2)

try:
    response = requests.get("http://localhost:8001/api/tasks", timeout=5)
    print(f"✅ Сервер работает! Статус: {response.status_code}")
    print("🌐 URL: http://localhost:8001")
    print("🎯 Откройте браузер и перейдите на http://localhost:8001")
except Exception as e:
    print(f"❌ Сервер не работает: {e}")
    print("🚀 Запустите: python3 main.py")
