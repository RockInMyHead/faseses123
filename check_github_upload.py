#!/usr/bin/env python3
"""
Проверка статуса загрузки на GitHub
"""

import os
import sys
import subprocess

def run_command(cmd, description=""):
    """Выполняет команду"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def main():
    print("CHECKING GITHUB UPLOAD STATUS")
    print("=" * 50)

    # Проверяем git статус
    print("Git status:")
    success, stdout, stderr = run_command("git status --porcelain")
    if success:
        if stdout.strip():
            print(f"Files to commit: {len(stdout.strip().split('\n'))} files")
            for line in stdout.strip().split('\n')[:10]:  # Показываем первые 10
                print(f"  {line}")
            if len(stdout.strip().split('\n')) > 10:
                print(f"  ... and {len(stdout.strip().split('\n')) - 10} more")
        else:
            print("No uncommitted changes")
    else:
        print(f"Error: {stderr}")

    # Проверяем remote
    print("\nRemote repositories:")
    success, stdout, stderr = run_command("git remote -v")
    if success and stdout.strip():
        for line in stdout.strip().split('\n'):
            print(f"  {line}")
    else:
        print("No remotes configured")

    # Проверяем последний коммит
    print("\nLast commit:")
    success, stdout, stderr = run_command("git log --oneline -3")
    if success and stdout.strip():
        for line in stdout.strip().split('\n'):
            print(f"  {line}")
    else:
        print("No commits found")

    # Проверяем статус репозитория
    print("\nRepository status:")
    success, stdout, stderr = run_command("git status -b --ahead-behind")
    if success and stdout.strip():
        for line in stdout.strip().split('\n'):
            print(f"  {line}")
    else:
        print("Could not get status")

    # Проверяем ветки
    print("\nBranches:")
    success, stdout, stderr = run_command("git branch -a")
    if success and stdout.strip():
        for line in stdout.strip().split('\n'):
            print(f"  {line}")
    else:
        print("No branches found")

    print("\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("1. If you see uncommitted files, run: git add . && git commit -m 'Update'")
    print("2. If no remote, run: git remote add origin https://github.com/RockInMyHead/face_relis_project.git")
    print("3. To push, run: git push -u origin main")
    print("4. Check repository: https://github.com/RockInMyHead/face_relis_project.git")

if __name__ == "__main__":
    main()
