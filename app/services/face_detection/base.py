"""
Base interface for face detection services
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol
from pathlib import Path
import numpy as np

from ...models.domain import Face


class FaceDetectionService(ABC):
    """
    Abstract base class for face detection and embedding services
    """

    @abstractmethod
    def detect_faces(self, image: np.ndarray, image_path: Path) -> List[Face]:
        """
        Detect faces in image and extract embeddings

        Args:
            image: Image as numpy array
            image_path: Path to original image file

        Returns:
            List of detected Face objects
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if face detection service is available"""
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """Get human-readable service name"""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the face detection model"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass


class QualityValidator(ABC):
    """
    Abstract base class for face quality validation
    """

    @abstractmethod
    def validate_face_quality(self, face: Face, image: np.ndarray) -> float:
        """
        Validate face quality

        Args:
            face: Face object to validate
            image: Original image

        Returns:
            Quality score from 0.0 to 1.0
        """
        pass
