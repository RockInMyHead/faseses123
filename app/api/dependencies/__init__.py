"""
Dependency injection container and providers
"""
from typing import Optional

from ...core.config import settings
from ...core.logging import get_logger
from ..services.file_service import FileService, LocalFileService
from ..services.face_detection import (
    FaceDetectionService,
    ArcFaceEmbedder,
    DualFaceEmbedder
)
from ..services.clustering import ClusteringService

logger = get_logger(__name__)

# Global service instances
_file_service: Optional[FileService] = None
_face_detection_service: Optional[FaceDetectionService] = None
_clustering_service: Optional[ClusteringService] = None


def get_file_service() -> FileService:
    """Get file service instance"""
    global _file_service
    if _file_service is None:
        _file_service = LocalFileService()
    return _file_service


def get_face_detection_service() -> Optional[FaceDetectionService]:
    """Get face detection service instance"""
    global _face_detection_service
    if _face_detection_service is None:
        try:
            if settings.use_dual_embedder:
                # Используем двойное распознавание
                _face_detection_service = DualFaceEmbedder(
                    primary_model=settings.insightface_model,
                    secondary_model=settings.secondary_model,
                    quality_threshold=settings.quality_threshold,
                    use_advanced_validation=settings.use_advanced_quality_validation
                )
                logger.info(f"Initialized DualFaceEmbedder: {settings.insightface_model} + {settings.secondary_model}")
            else:
                # Используем обычный эмбеддер
                from ..services.face_detection import ArcFaceConfig
                config = ArcFaceConfig()
                _face_detection_service = ArcFaceEmbedder(config, settings.insightface_model)
                logger.info(f"Initialized ArcFaceEmbedder: {settings.insightface_model}")

            # Инициализация сервиса
            _face_detection_service.initialize()

        except Exception as e:
            logger.error(f"Failed to initialize face detection service: {e}")
            return None

    return _face_detection_service


def get_clustering_service() -> Optional[ClusteringService]:
    """Get clustering service instance"""
    global _clustering_service
    if _clustering_service is None:
        # TODO: Initialize clustering service
        pass
    return _clustering_service


def reset_services():
    """Reset all service instances (useful for testing)"""
    global _file_service, _face_detection_service, _clustering_service
    _file_service = None
    _face_detection_service = None
    _clustering_service = None
    logger.info("All services reset")