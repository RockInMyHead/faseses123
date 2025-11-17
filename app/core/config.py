"""
Application configuration settings
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Face Detection Settings
    insightface_model: str = "buffalo_l"
    secondary_model: str = "buffalo_m"
    use_dual_embedder: bool = True  # Использовать двойное распознавание
    use_advanced_quality_validation: bool = True  # Продвинутая валидация качества
    use_advanced_rescue: bool = True  # Использовать продвинутую систему rescue
    rescue_strategy: str = "balanced"  # Стратегия rescue: conservative, balanced, aggressive, context_aware
    quality_threshold: float = 0.75
    min_face_size: int = 64
    max_face_size: int = 512
    max_face_angle: int = 30

    # Clustering Settings
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: Optional[int] = None

    # File System Settings
    supported_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    max_image_size_mb: int = 50

    # Task Management
    max_concurrent_tasks: int = 3
    task_timeout_seconds: int = 3600  # 1 hour

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
