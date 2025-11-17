"""
Base interface for file system operations
"""
from abc import ABC, abstractmethod
from typing import List, Protocol, Optional
from pathlib import Path

from ...models.domain import FolderInfo


class ProgressCallback(Protocol):
    """Protocol for progress callback functions"""
    def __call__(self, message: str, progress: Optional[int] = None) -> None:
        ...


class FileService(ABC):
    """
    Abstract base class for file system operations
    """

    @abstractmethod
    async def get_folder_contents(self, path: Path) -> List[FolderInfo]:
        """
        Get contents of directory

        Args:
            path: Directory path

        Returns:
            List of folder/file information
        """
        pass

    @abstractmethod
    async def create_folder(self, path: Path, name: str) -> Path:
        """
        Create new folder

        Args:
            path: Parent directory path
            name: New folder name

        Returns:
            Path to created folder
        """
        pass

    @abstractmethod
    async def delete_item(self, path: Path) -> None:
        """
        Delete file or folder

        Args:
            path: Path to delete
        """
        pass

    @abstractmethod
    async def move_item(self, source: Path, destination: Path) -> None:
        """
        Move file or folder

        Args:
            source: Source path
            destination: Destination path
        """
        pass

    @abstractmethod
    async def copy_item(self, source: Path, destination: Path) -> None:
        """
        Copy file or folder

        Args:
            source: Source path
            destination: Destination path
        """
        pass

    @abstractmethod
    def get_image_preview(self, path: Path, size: int = 150) -> Optional[str]:
        """
        Get base64 encoded image preview

        Args:
            path: Image file path
            size: Preview size

        Returns:
            Base64 encoded image or None if failed
        """
        pass
