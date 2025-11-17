"""
Pytest configuration and fixtures
"""
import pytest
from pathlib import Path
from unittest.mock import Mock

from app.core.config import Settings
from app.services.file_service.local import LocalFileService


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create temporary directory for testing"""
    return tmp_path


@pytest.fixture
def sample_image_dir(temp_dir: Path) -> Path:
    """Create directory with sample image files"""
    img_dir = temp_dir / "images"
    img_dir.mkdir()

    # Create some dummy image files
    for i in range(3):
        (img_dir / f"test_image_{i}.jpg").write_text("fake image content")

    return img_dir


@pytest.fixture
def file_service() -> LocalFileService:
    """Get file service instance"""
    return LocalFileService()


@pytest.fixture
def mock_settings() -> Settings:
    """Mock settings for testing"""
    settings = Mock(spec=Settings)
    settings.supported_extensions = (".jpg", ".jpeg", ".png")
    settings.max_image_size_mb = 50
    return settings