"""
Local file system service implementation
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional

from ...core.logging import get_logger
from ...models.domain import FolderInfo
from ...utils.file_utils import is_supported_image
from ...utils.image_utils import load_image_safe, resize_image_for_preview, image_to_base64
from .base import FileService

logger = get_logger(__name__)


class LocalFileService(FileService):
    """Local file system operations service"""

    async def get_folder_contents(self, path: Path) -> List[FolderInfo]:
        """Get folder contents with metadata"""
        try:
            contents = []

            for item in path.iterdir():
                try:
                    stat = item.stat()

                    folder_info = FolderInfo(
                        name=item.name,
                        path=str(item),
                        type="directory" if item.is_dir() else "file",
                        size=stat.st_size if item.is_file() else None,
                        modified=stat.st_mtime,
                        children_count=len(list(item.iterdir())) if item.is_dir() else None
                    )
                    contents.append(folder_info)

                except (OSError, PermissionError) as e:
                    logger.warning(f"Failed to get info for {item}: {e}")
                    continue

            # Sort: directories first, then files, alphabetically
            contents.sort(key=lambda x: (x.type != "directory", x.name.lower()))

            return contents

        except Exception as e:
            logger.error(f"Failed to get folder contents for {path}: {e}")
            return []

    async def create_folder(self, path: Path, name: str) -> Path:
        """Create new folder"""
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid parent directory: {path}")

        new_folder = path / name
        new_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created folder: {new_folder}")
        return new_folder

    async def delete_item(self, path: Path) -> None:
        """Delete file or folder"""
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            path.unlink()
            logger.info(f"Deleted file: {path}")
        else:
            shutil.rmtree(path)
            logger.info(f"Deleted directory: {path}")

    async def move_item(self, source: Path, destination: Path) -> None:
        """Move file or folder"""
        if not source.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")

        # Ensure destination parent exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(source), str(destination))
        logger.info(f"Moved {source} to {destination}")

    async def copy_item(self, source: Path, destination: Path) -> None:
        """Copy file or folder"""
        if not source.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")

        # Ensure destination parent exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        if source.is_file():
            shutil.copy2(source, destination)
        else:
            shutil.copytree(source, destination, dirs_exist_ok=True)

        logger.info(f"Copied {source} to {destination}")

    def get_image_preview(self, path: Path, size: int = 150) -> Optional[str]:
        """Get base64 encoded image preview"""
        try:
            if not path.exists() or not path.is_file():
                return None

            if not is_supported_image(path):
                return None

            # Load and resize image
            image = load_image_safe(path)
            if image is None:
                return None

            resized = resize_image_for_preview(image, size)

            # Convert to base64
            return image_to_base64(resized)

        except Exception as e:
            logger.warning(f"Failed to generate preview for {path}: {e}")
            return None
