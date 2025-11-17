"""
Base interface for clustering services
"""
from abc import ABC, abstractmethod
from typing import List, Protocol, Optional
from pathlib import Path

from ...models.domain import ClusteringResult, Face


class ProgressCallback(Protocol):
    """Protocol for progress callback functions"""
    def __call__(self, message: str, progress: Optional[int] = None) -> None:
        ...


class ClusteringService(ABC):
    """
    Abstract base class for face clustering services
    """

    @abstractmethod
    async def cluster_faces(
        self,
        faces: List[Face],
        progress_callback: Optional[ProgressCallback] = None
    ) -> ClusteringResult:
        """
        Perform face clustering

        Args:
            faces: List of detected faces
            progress_callback: Optional callback for progress updates

        Returns:
            ClusteringResult with clusters and metrics
        """
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """Get human-readable service name"""
        pass

    @abstractmethod
    def get_service_type(self) -> str:
        """Get service type ('local' or 'global')"""
        pass
