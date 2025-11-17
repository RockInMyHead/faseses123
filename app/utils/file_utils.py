"""
File system utility functions
"""
import os
from pathlib import Path
from typing import List, Optional, Set

from ..core.config import settings


def is_supported_image(path: Path) -> bool:
    """Check if file is a supported image format"""
    return path.suffix.lower() in settings.supported_extensions


def find_images_in_directory(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Find all supported images in directory

    Args:
        directory: Directory to search in
        recursive: Whether to search recursively

    Returns:
        List of image file paths
    """
    if not directory.exists() or not directory.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    images = []

    for ext in settings.supported_extensions:
        images.extend(directory.glob(f"{pattern}{ext}"))
        images.extend(directory.glob(f"{pattern}{ext.upper()}"))

    return sorted(images)


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes"""
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def validate_image_size(path: Path) -> bool:
    """Validate that image size is within acceptable limits"""
    return get_file_size_mb(path) <= settings.max_image_size_mb


def get_logical_drives() -> List[Dict[str, str]]:
    """Get list of available logical drives"""
    drives = []
    if os.name == 'nt':  # Windows
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append({
                    "name": f"Drive {letter}",
                    "path": drive
                })
    else:  # Unix-like systems
        drives.append({
            "name": "Root",
            "path": "/"
        })

    return drives


def get_special_dirs() -> Dict[str, str]:
    """Get special system directories"""
    special_dirs = {}

    if os.name == 'nt':  # Windows
        special_dirs.update({
            "Desktop": os.path.join(os.path.expanduser("~"), "Desktop"),
            "Documents": os.path.join(os.path.expanduser("~"), "Documents"),
            "Pictures": os.path.join(os.path.expanduser("~"), "Pictures"),
            "Downloads": os.path.join(os.path.expanduser("~"), "Downloads"),
        })
    else:  # Unix-like systems
        special_dirs.update({
            "Home": os.path.expanduser("~"),
            "Desktop": os.path.join(os.path.expanduser("~"), "Desktop"),
            "Pictures": os.path.join(os.path.expanduser("~"), "Pictures"),
            "Downloads": os.path.join(os.path.expanduser("~"), "Downloads"),
        })

    # Filter out non-existent directories
    return {name: path for name, path in special_dirs.items() if os.path.exists(path)}
