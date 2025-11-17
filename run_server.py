#!/usr/bin/env python3
"""
Скрипт запуска сервера FaceSort
"""

import subprocess
import sys
import time
import os
import signal

def start_server():
    """Запускает сервер FaceSort"""
    print("🚀 Запуск проекта FaceSort...")
    print(f"📁 Рабочая директория: {os.getcwd()}")
    print(f"🐍 Python версия: {sys.version}")
    print("")
    
    # Останавливаем предыдущие процессы
    print("🛑 Останавливаем предыдущие процессы...")
    try:
        subprocess.run(["pkill", "-f", "python.*main.py"], capture_output=True)
        time.sleep(2)
    except:
        pass
    
    # Запускаем сервер
    print("🚀 Запускаем сервер FaceSort...")
    try:
        process = subprocess.Popen([sys.executable, "main.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        print(f"✅ Сервер запущен с PID: {process.pid}")
        print("🌐 URL: http://localhost:8000")
        print("📊 Проверка через 5 секунд...")
        
        time.sleep(5)
        
        # Проверяем, что сервер работает
        try:
            import requests
            response = requests.get("http://localhost:8000/api/tasks", timeout=5)
            if response.status_code == 200:
                print("✅ Сервер работает!")
                print("🎯 Откройте http://localhost:8000 в браузере")
                print("🔧 Используйте кнопку '📂 Общие' для тестирования")
                print("")
                print("📋 Для остановки сервера нажмите Ctrl+C")
                
                # Ждем завершения
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Останавливаем сервер...")
                    process.terminate()
                    process.wait()
                    print("✅ Сервер остановлен")
            else:
                print(f"❌ Сервер отвечает с кодом: {response.status_code}")
        except ImportError:
            print("⚠️ Модуль requests не установлен, пропускаем проверку")
            print("🎯 Откройте http://localhost:8000 в браузере")
            process.wait()
        except Exception as e:
            print(f"❌ Ошибка проверки сервера: {e}")
            print("🎯 Попробуйте открыть http://localhost:8000 в браузере")
            process.wait()
            
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()
