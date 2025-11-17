#!/usr/bin/env python3
import subprocess
import time
import os
import signal

print("🔄 Перезапускаем сервер FaceSort...")

# Останавливаем все процессы Python
try:
    subprocess.run(["pkill", "-f", "python.*main.py"], check=False)
    print("✅ Остановлены предыдущие процессы")
except:
    print("⚠️ Не удалось остановить процессы")

time.sleep(2)

# Запускаем сервер на порту 8001
print("🚀 Запускаем сервер на порту 8001...")
try:
    subprocess.Popen(["python3", "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("✅ Сервер запущен")
    print("🌐 URL: http://localhost:8001")
    print("🎯 Откройте браузер и перейдите на http://localhost:8001")
except Exception as e:
    print(f"❌ Ошибка запуска сервера: {e}")
