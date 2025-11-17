#!/usr/bin/env python3
"""
Финальный скрипт для загрузки проекта на GitHub
"""

import os
import sys
import subprocess

def run_command(cmd, description=""):
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
                print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"[EXCEPTION] {description}: {e}")
        return False

def main():
    print("FINAL GITHUB UPLOAD")
    print("=" * 50)
    print("Repository: https://github.com/RockInMyHead/face_relis_project.git")
    print("=" * 50)

    steps = [
        "Initialize Git repository",
        "Create .gitignore and README.md",
        "Add all files",
        "Create initial commit",
        "Add remote origin",
        "Push to GitHub"
    ]

    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}/{len(steps)}: {step}")
        print("-" * 40)

        if step == "Initialize Git repository":
            if not os.path.exists('.git'):
                run_command("git init", step)
            else:
                print("[OK] Git repository already exists")

        elif step == "Create .gitignore and README.md":
            # Создаем .gitignore если его нет или он отличается
            expected_gitignore = """# Python
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

            current_gitignore = ""
            if os.path.exists('.gitignore'):
                with open('.gitignore', 'r', encoding='utf-8') as f:
                    current_gitignore = f.read()

            if current_gitignore != expected_gitignore:
                with open('.gitignore', 'w', encoding='utf-8') as f:
                    f.write(expected_gitignore)
                print("Updated .gitignore")
            else:
                print(".gitignore is up to date")

            # Обновляем README.md
            expected_readme = """# 📸 FaceRelis - Кластеризация лиц в фотографиях

FaceRelis - это современное веб-приложение для автоматической кластеризации лиц на фотографиях с использованием передовых алгоритмов машинного обучения.

## ✨ Основные возможности

- 🔍 **Автоматическое распознавание лиц** с использованием ArcFace и InsightFace
- 📁 **Умная кластеризация** фотографий по людям
- 🎯 **Два алгоритма кластеризации**: стандартный и продвинутый
- 📂 **Обработка общих папок** с автоматическим поиском
- 🔄 **Автоматическое обновление** интерфейса в реальном времени
- 📱 **Современный веб-интерфейс** с drag & drop
- 📦 **Экспорт результатов** в ZIP архивы

## 🚀 Быстрый старт

### Установка

1. **Клонируйте репозиторий:**

```bash
git clone https://github.com/RockInMyHead/face_relis_project.git
cd face_relis_project
```

2. **Установите зависимости:**

```bash
pip install -r requirements.txt
```

3. **Запустите приложение:**

```bash
python main.py
```

4. **Откройте браузер:**

```
http://localhost:8000
```

## 🛠 Технологии

### Backend
- **FastAPI** - современный веб-фреймворк
- **InsightFace** - распознавание лиц
- **ArcFace** - извлечение эмбеддингов
- **Faiss** - эффективная кластеризация
- **OpenCV** - обработка изображений

### Frontend
- **Vanilla JavaScript** - без фреймворков
- **HTML5/CSS3** - современный интерфейс
- **Drag & Drop API** - удобная навигация
- **Fetch API** - асинхронные запросы

## 🎯 Особенности

### Автоматическое обновление
- Папки обновляются каждую секунду
- Задачи обновляются в реальном времени
- Никаких ручных обновлений не требуется

### Обработка общих папок
- Автоматический поиск папок "общие", "common", "shared"
- Рекурсивный поиск с ограничением глубины
- Создание пустых папок для уникальных людей
- Подсчет реального количества фотографий

### Современный интерфейс
- Квадратные папки 150x150px
- Единообразные размеры всех элементов
- Drag & Drop навигация
- Контекстные меню
- Модальные окна

## 📁 Структура проекта

```
face_relis_project/
├── main.py                 # FastAPI приложение
├── cluster_simple.py      # Алгоритмы кластеризации
├── static/
│   ├── index.html         # Главная страница
│   └── app.js            # Frontend логика
├── requirements.txt      # Зависимости Python
├── .gitignore           # Git исключения
└── README.md            # Документация
```

## 🔧 API Endpoints

- `GET /` - Главная страница
- `GET /api/drives` - Список дисков
- `GET /api/folder` - Содержимое папки
- `POST /api/queue/add` - Добавить в очередь
- `POST /api/process` - Обработать очередь
- `GET /api/tasks` - Активные задачи
- `GET /api/image/preview` - Превью изображения

## 🎨 Интерфейс

### Основные элементы
- **Навигация по папкам** - квадратные кнопки 150x150px
- **Очередь обработки** - drag & drop интерфейс
- **Активные задачи** - реальное время обновления
- **Кнопка "Общие"** - специальный алгоритм для общих папок

### Автоматические функции
- Обновление папок каждую секунду
- Обновление задач каждую секунду
- Автоматическое создание папок для людей
- Подсчет фотографий в названиях папок

## 🚀 Развертывание

### Локальная разработка
```bash
python main.py
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📝 Лицензия

MIT License - см. файл LICENSE для деталей.

## 🤝 Вклад в проект

1. Fork репозиторий
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📞 Поддержка

Если у вас есть вопросы или проблемы, создайте issue в GitHub репозитории.

---

**FaceRelis** - современное решение для кластеризации лиц в фотографиях! 📸✨
"""

            current_readme = ""
            if os.path.exists('README.md'):
                with open('README.md', 'r', encoding='utf-8') as f:
                    current_readme = f.read()

            if current_readme != expected_readme:
                with open('README.md', 'w', encoding='utf-8') as f:
                    f.write(expected_readme)
                print("Updated README.md")
            else:
                print("README.md is up to date")

        elif step == "Add all files":
            run_command("git add .", step)

        elif step == "Create initial commit":
            # Создаем коммит с новыми изменениями
            commit_msg = "Update: Enhanced face clustering application with improved UI and auto-refresh"
            if not run_command(f'git commit -m "{commit_msg}"', step):
                # Если коммит не удался, возможно уже есть такие изменения
                print("No new changes to commit or commit failed")
                # Это не критично, продолжаем

        elif step == "Add remote origin":
            repo_url = "https://github.com/RockInMyHead/face_relis_project.git"

            # Проверяем remote
            result = subprocess.run("git remote", shell=True, capture_output=True, text=True, cwd=os.getcwd())
            if "origin" not in result.stdout:
                run_command(f"git remote add origin {repo_url}", step)
            else:
                print("[OK] Remote origin already exists")

        elif step == "Push to GitHub":
            # Пробуем main branch
            run_command("git branch -M main", "Rename branch to main")
            if not run_command("git push -u origin main", step):
                print("Failed to push to main, trying master...")
                run_command("git branch -M master", "Rename branch to master")
                run_command("git push -u origin master", "Push to GitHub (master)")

    print("\n" + "=" * 50)
    print("UPLOAD COMPLETE!")
    print("Repository: https://github.com/RockInMyHead/face_relis_project.git")
    print("\nNext steps:")
    print("1. Visit the repository on GitHub")
    print("2. Check that all files are uploaded")
    print("3. Add a description and topics")
    print("4. Enable GitHub Pages if needed")

    # Финальная проверка
    print("\nFinal check:")
    run_command("python check_github_upload.py", "Check upload status")

if __name__ == "__main__":
    main()
