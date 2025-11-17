"""
Image processing utility functions
"""
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps
import numpy as np

from ..core.logging import get_logger

logger = get_logger(__name__)


def load_image_safe(path: Path) -> Optional[np.ndarray]:
    """
    Safely load image from file

    Args:
        path: Path to image file

    Returns:
        Image as numpy array or None if loading failed
    """
    try:
        with Image.open(path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Return as numpy array
            return np.array(img)
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return None


def resize_image_for_preview(image: np.ndarray, max_size: int = 150) -> np.ndarray:
    """
    Resize image for preview while maintaining aspect ratio

    Args:
        image: Input image as numpy array
        max_size: Maximum dimension size

    Returns:
        Resized image
    """
    pil_image = Image.fromarray(image)

    # Calculate new size maintaining aspect ratio
    width, height = pil_image.size
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    # Resize image
    resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(resized)


def image_to_base64(image: np.ndarray, format: str = 'JPEG') -> str:
    """
    Convert numpy image array to base64 string

    Args:
        image: Image as numpy array
        format: Image format ('JPEG', 'PNG', etc.)

    Returns:
        Base64 encoded image string
    """
    pil_image = Image.fromarray(image)

    # Save to bytes buffer
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)

    # Convert to base64
    return base64.b64encode(buffer.getvalue()).decode()


def calculate_image_hash(image: np.ndarray) -> str:
    """
    Calculate perceptual hash of image for duplicate detection

    Args:
        image: Image as numpy array

    Returns:
        Image hash string
    """
    try:
        pil_image = Image.fromarray(image)

        # Convert to grayscale and resize to 8x8 for hash
        gray = ImageOps.grayscale(pil_image)
        small = gray.resize((8, 8), Image.Resampling.LANCZOS)

        # Calculate average pixel value
        pixels = list(small.getdata())
        avg = sum(pixels) / len(pixels)

        # Create binary hash
        bits = "".join(['1' if pixel > avg else '0' for pixel in pixels])

        # Convert to hex
        return hex(int(bits, 2))[2:].zfill(16)

    except Exception as e:
        logger.warning(f"Failed to calculate image hash: {e}")
        return ""


def get_image_dimensions(path: Path) -> Optional[Tuple[int, int]]:
    """
    Get image dimensions without loading full image

    Args:
        path: Path to image file

    Returns:
        Tuple of (width, height) or None if failed
    """
    try:
        with Image.open(path) as img:
            return img.size
    except Exception as e:
        logger.warning(f"Failed to get image dimensions for {path}: {e}")
        return None
