"""
Unit tests for file service
"""
import pytest
from pathlib import Path
from unittest.mock import patch

from app.services.file_service.local import LocalFileService


class TestLocalFileService:
    """Test LocalFileService functionality"""

    @pytest.mark.asyncio
    async def test_get_folder_contents(self, temp_dir: Path, file_service: LocalFileService):
        """Test getting folder contents"""
        # Create some test files and directories
        (temp_dir / "file1.txt").write_text("content")
        (temp_dir / "file2.jpg").write_text("image content")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "subfile.txt").write_text("sub content")

        contents = await file_service.get_folder_contents(temp_dir)

        # Should have 3 items: 2 files + 1 directory
        assert len(contents) == 3

        # Check that directory comes first
        assert contents[0].name == "subdir"
        assert contents[0].type == "directory"

        # Check file items
        file_names = [item.name for item in contents if item.type == "file"]
        assert "file1.txt" in file_names
        assert "file2.jpg" in file_names

    @pytest.mark.asyncio
    async def test_create_folder(self, temp_dir: Path, file_service: LocalFileService):
        """Test folder creation"""
        new_folder = await file_service.create_folder(temp_dir, "test_folder")

        assert new_folder.exists()
        assert new_folder.is_dir()
        assert new_folder.name == "test_folder"

    @pytest.mark.asyncio
    async def test_create_folder_invalid_parent(self, file_service: LocalFileService):
        """Test folder creation with invalid parent"""
        with pytest.raises(ValueError):
            await file_service.create_folder(Path("/nonexistent/path"), "test")

    def test_get_image_preview_nonexistent(self, file_service: LocalFileService):
        """Test image preview for nonexistent file"""
        result = file_service.get_image_preview(Path("/nonexistent/image.jpg"))
        assert result is None

    def test_get_image_preview_unsupported_format(self, temp_dir: Path, file_service: LocalFileService):
        """Test image preview for unsupported format"""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("not an image")

        result = file_service.get_image_preview(txt_file)
        assert result is None
