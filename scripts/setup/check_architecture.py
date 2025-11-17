#!/usr/bin/env python3
"""
Check that the new architecture is properly set up
"""
import sys
from pathlib import Path
from typing import List

def check_file_exists(path: Path, description: str) -> bool:
    """Check if file exists and print status"""
    exists = path.exists()
    status = "[OK]" if exists else "[FAIL]"
    print(f"{status} {description}: {path}")
    return exists

def check_package_imports(package_path: Path, modules: List[str]) -> bool:
    """Check if package modules can be imported"""
    success = True
    for module in modules:
        try:
            __import__(f"app.{module.replace('/', '.')}")
            print(f"[OK] Import {module}")
        except ImportError as e:
            print(f"[FAIL] Import {module}: {e}")
            success = False
    return success

def main():
    """Main check function"""
    print("Checking FaceRelis Clean Architecture Setup")
    print("=" * 50)

    root = Path(__file__).parent.parent.parent
    all_good = True

    # Check directory structure
    print("\nDirectory Structure:")
    directories = [
        ("app", "Main application package"),
        ("app/core", "Core functionality"),
        ("app/api", "API layer"),
        ("app/api/routes", "HTTP routes"),
        ("app/api/dependencies", "DI container"),
        ("app/services", "Business services"),
        ("app/services/file_service", "File operations"),
        ("app/services/face_detection", "Face detection"),
        ("app/services/clustering", "Clustering services"),
        ("app/models", "Data models"),
        ("app/utils", "Utilities"),
        ("tests", "Test suite"),
        ("tests/unit", "Unit tests"),
        ("tests/integration", "Integration tests"),
    ]

    for dir_path, description in directories:
        all_good &= check_file_exists(root / dir_path, description)

    # Check key files
    print("\nKey Files:")
    files = [
        ("app/__init__.py", "Main package init"),
        ("app/main.py", "FastAPI application"),
        ("app/core/config.py", "Configuration"),
        ("app/core/logging.py", "Logging setup"),
        ("app/core/exceptions.py", "Custom exceptions"),
        ("app/models/domain.py", "Domain models"),
        ("app/models/schemas.py", "API schemas"),
        ("app/services/file_service/local.py", "File service implementation"),
        ("app/api/routes/files.py", "File API routes"),
        ("app/api/dependencies/__init__.py", "DI container"),
        ("tests/conftest.py", "Test configuration"),
        ("main_new.py", "New application entry point"),
        ("ARCHITECTURE.md", "Architecture documentation"),
    ]

    for file_path, description in files:
        all_good &= check_file_exists(root / file_path, description)

    # Check imports
    print("\nPackage Imports:")
    modules_to_check = [
        "core.config",
        "core.logging",
        "core.exceptions",
        "models.domain",
        "models.schemas",
        "services.file_service.local",
        "utils.file_utils",
        "utils.image_utils",
    ]

    # Add app to path for import check
    sys.path.insert(0, str(root))
    try:
        all_good &= check_package_imports(root / "app", modules_to_check)
    finally:
        sys.path.pop(0)

    # Final result
    print("\n" + "=" * 50)
    if all_good:
        print("SUCCESS: Architecture check PASSED! Ready to run.")
        print("\nTo start the new application:")
        print("   python main_new.py")
        print("\nTo run tests:")
        print("   pytest tests/")
        return 0
    else:
        print("FAILED: Architecture check FAILED! Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
