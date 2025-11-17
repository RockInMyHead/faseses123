#!/usr/bin/env python3
"""
Создает фиктивный коммит для загрузки на GitHub
"""

import os
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
    print("CREATING DUMMY COMMIT FOR GITHUB UPLOAD")
    print("=" * 50)

    # Создаем фиктивный файл для коммита
    dummy_file = ".github_upload_marker"
    with open(dummy_file, 'w', encoding='utf-8') as f:
        f.write(f"Upload marker - {os.environ.get('USERNAME', 'unknown')} - {os.environ.get('COMPUTERNAME', 'unknown')}\n")
        f.write("FaceRelis Face Clustering Application\n")
        f.write("Uploaded from local development environment\n")

    print("Created dummy file for commit")

    # Добавляем файл
    run_command("git add .", "Add files")

    # Создаем коммит
    commit_msg = "Upload: FaceRelis face clustering application with enhanced features"
    if run_command(f'git commit -m "{commit_msg}"', "Create commit"):
        print("Commit created successfully")

        # Проверяем remote
        result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if "origin" not in result.stdout:
            run_command("git remote add origin https://github.com/RockInMyHead/face_relis_project.git", "Add remote")

        # Push
        run_command("git push -u origin main", "Push to main")
        if not run_command("git push -u origin main", "Push to main"):
            run_command("git push -u origin master", "Push to master")

    else:
        print("Failed to create commit")

    print("\n" + "=" * 50)
    print("Check: https://github.com/RockInMyHead/face_relis_project.git")

if __name__ == "__main__":
    main()
