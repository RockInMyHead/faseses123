#!/usr/bin/env python3
"""
Скрипт для загрузки проекта на GitHub
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Выполняет команду"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"[OK] {description}")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            print(f"[FAILED] {description}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
    except Exception as e:
        print(f"[EXCEPTION] {description}: {e}")
        return False

def setup_git_repo():
    """Настраивает git репозиторий"""
    print("Setting up Git repository...")
    print("=" * 50)

    # Инициализируем git если не инициализирован
    if not os.path.exists('.git'):
        print("Initializing Git repository...")
        if not run_command("git init", "Initialize Git repository"):
            return False
    else:
        print("Git repository already exists")

    # Создаем .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary files
*.tmp
*.temp

# ML Models (downloaded automatically)
.insightface/
*.onnx
*.pb
*.h5

# Test images (keep some for demo)
# Uncomment to ignore all images:
# *.jpg
# *.png
# *.jpeg
# *.bmp
# *.tiff

# Keep demo images
!test_photos/
test_photos/*/
!test_photos/*.jpg
!test_photos/*.png

# Large files
*.zip
*.tar.gz
*.7z

# Database files
*.db
*.sqlite
*.sqlite3
"""

    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)

    print("Created .gitignore")

    # Добавляем файлы
    if not run_command("git add .", "Add all files"):
        return False

    # Создаем README.md если его нет
    if not os.path.exists('README.md'):
        readme_content = """# Face Clustering Application

Автоматическая кластеризация фотографий по лицам с использованием AI.

## Особенности

- Распознавание лиц с помощью InsightFace (ArcFace)
- Кластеризация с использованием Faiss
- Веб-интерфейс для управления
- Поддержка различных моделей распознавания
- Групповая обработка общих фотографий

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/RockInMyHead/face_0711.git
cd face_0711
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите сервер:
```bash
python main.py
```

4. Откройте браузер: http://localhost:8000

## Использование

1. Выберите папку с фотографиями
2. Добавьте папку в очередь обработки
3. Нажмите "Обработать очередь"
4. Просматривайте прогресс в разделе "Активные задачи"

## Модели распознавания

- **InsightFace (buffalo_l)** - максимальная точность (рекомендуется)
- **InsightFace (buffalo_s)** - быстрая и легкая
- **Face Recognition (dlib)** - запасной вариант

## Структура проекта

- `main.py` - основной сервер FastAPI
- `cluster_simple.py` - кластеризация с InsightFace
- `cluster_face_recognition.py` - кластеризация с face_recognition
- `static/` - веб-интерфейс
- `requirements.txt` - зависимости Python

## Лицензия

MIT License
"""

        with open('README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print("Created README.md")

    # Коммитим изменения
    if not run_command('git commit -m "Initial commit: Face clustering application"', "Create initial commit"):
        # Если коммит не удался (возможно, уже есть коммиты), попробуем добавить файлы заново
        run_command("git add .", "Re-add files")
        run_command('git commit -m "Update: Face clustering application"', "Create update commit")

    return True

def upload_to_github():
    """Загружает на GitHub"""
    print("\nUploading to GitHub...")
    print("=" * 50)

    repo_url = "https://github.com/RockInMyHead/face_relis_project.git"

    # Проверяем remote
    result = subprocess.run("git remote", shell=True, capture_output=True, text=True, cwd=os.getcwd())
    if "origin" not in result.stdout:
        if not run_command(f"git remote add origin {repo_url}", "Add remote origin"):
            return False
    else:
        # Проверяем правильный URL
        result = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0 and repo_url not in result.stdout:
            run_command("git remote remove origin", "Remove old remote")
            run_command(f"git remote add origin {repo_url}", "Add correct remote")

    # Пушим на GitHub
    # Сначала пробуем main branch (новый стандарт GitHub)
    run_command("git branch -M main", "Rename branch to main")
    if not run_command("git push -u origin main", "Push to GitHub (main)"):
        print("Failed to push to main, trying master...")
        run_command("git branch -M master", "Rename branch to master")
        run_command("git push -u origin master", "Push to GitHub (master)")

    return True

def main():
    print("GIT UPLOAD TO GITHUB")
    print("=" * 50)
    print("Repository: https://github.com/RockInMyHead/face_0711.git")
    print("=" * 50)

    # Настраиваем git
    if not setup_git_repo():
        print("\n[FAILED] Failed to setup Git repository")
        return False

    # Загружаем на GitHub
    if not upload_to_github():
        print("\n[FAILED] Failed to upload to GitHub")
        return False

    print("\n[SUCCESS] Project uploaded to GitHub!")
    print("Repository: https://github.com/RockInMyHead/face_0711.git")
    print("\nNext steps:")
    print("1. Check the repository on GitHub")
    print("2. Enable GitHub Pages if needed")
    print("3. Add collaborators if needed")

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n[FAILED] Upload failed. Check the errors above.")
        sys.exit(1)
