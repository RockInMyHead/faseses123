"""
Face detection and embedding services
"""
from .base import FaceDetectionService
from .embedder import ArcFaceEmbedder, ArcFaceConfig
from .dual_embedder import DualFaceEmbedder
from .dual_quality_validator import (
    DualQualityValidator,
    QualityMethod,
    QualityMetrics,
    ValidationResult
)
from .advanced_rescue import (
    AdvancedFaceRescue,
    RescueStrategy,
    RescueCandidate,
    RescueResult
)

__all__ = [
    "FaceDetectionService",
    "ArcFaceEmbedder",
    "ArcFaceConfig",
    "DualFaceEmbedder",
    "DualQualityValidator",
    "QualityMethod",
    "QualityMetrics",
    "ValidationResult",
    "AdvancedFaceRescue",
    "RescueStrategy",
    "RescueCandidate",
    "RescueResult"
]
