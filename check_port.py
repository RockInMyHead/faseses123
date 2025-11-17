#!/usr/bin/env python3
import socket
import time

def check_port(port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False

print("🔍 Проверяем порты...")

# Проверяем порт 8000
if check_port(8000):
    print("✅ Порт 8000: ЗАНЯТ")
else:
    print("❌ Порт 8000: СВОБОДЕН")

# Проверяем порт 8001
if check_port(8001):
    print("✅ Порт 8001: ЗАНЯТ")
else:
    print("❌ Порт 8001: СВОБОДЕН")

print("\n🎯 Если порт 8001 занят, сервер работает!")
print("🌐 Откройте браузер: http://localhost:8001")
