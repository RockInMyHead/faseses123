#!/usr/bin/env python3
"""
Принудительная загрузка на GitHub
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
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return False
    except Exception as e:
        print(f"[EXCEPTION] {description}: {e}")
        return False

def main():
    print("FORCE UPLOAD TO GITHUB")
    print("=" * 50)
    print("Repository: https://github.com/RockInMyHead/face_relis_project.git")
    print("=" * 50)

    # Проверяем статус
    print("\nChecking git status...")
    run_command("git status --porcelain", "Check status")

    # Принудительно добавляем все файлы
    print("\nForcing add all files...")
    run_command("git add -A", "Force add all files")

    # Проверяем статус снова
    print("\nChecking git status after add...")
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True, cwd=os.getcwd())
    if result.stdout.strip():
        print(f"Files to commit: {len(result.stdout.strip().split('\n'))} files")
        for line in result.stdout.strip().split('\n')[:10]:
            print(f"  {line}")

        # Создаем принудительный коммит
        print("\nCreating forced commit...")
        commit_msg = "Force update: FaceRelis face clustering application with enhanced features"
        if run_command(f'git commit -m "{commit_msg}" --allow-empty', "Create commit"):
            print("Commit created successfully")
        else:
            print("Trying amend...")
            run_command(f'git commit --amend -m "{commit_msg}"', "Amend commit")
    else:
        print("No changes to commit")
        # Создаем пустой коммит
        print("\nCreating empty commit...")
        run_command('git commit --allow-empty -m "Empty commit: Force push"', "Empty commit")

    # Проверяем remote
    print("\nChecking remote...")
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True, cwd=os.getcwd())
    if "origin" in result.stdout:
        print("Remote origin exists")
        # Проверяем URL
        result = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if "face_relis_project" not in result.stdout:
            print("Wrong remote URL, updating...")
            run_command("git remote set-url origin https://github.com/RockInMyHead/face_relis_project.git", "Update remote URL")
    else:
        print("Adding remote origin...")
        run_command("git remote add origin https://github.com/RockInMyHead/face_relis_project.git", "Add remote")

    # Принудительный push
    print("\nForce pushing to GitHub...")
    print("Trying main branch...")
    if not run_command("git push -u origin main --force", "Force push main"):
        print("Trying master branch...")
        run_command("git branch -M master", "Rename to master")
        run_command("git push -u origin master --force", "Force push master")

    # Финальная проверка
    print("\nFinal status:")
    run_command("git log --oneline -3", "Recent commits")
    run_command("git status", "Final status")

    print("\n" + "=" * 50)
    print("UPLOAD COMPLETE!")
    print("Check: https://github.com/RockInMyHead/face_relis_project.git")
    print("=" * 50)

if __name__ == "__main__":
    main()
