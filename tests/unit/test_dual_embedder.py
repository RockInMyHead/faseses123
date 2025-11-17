"""
Unit tests for DualFaceEmbedder
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.face_detection.dual_embedder import DualFaceEmbedder
from app.core.config import Settings


class TestDualFaceEmbedder:
    """Test DualFaceEmbedder functionality"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing"""
        settings = Mock(spec=Settings)
        settings.insightface_model = "buffalo_l"
        settings.secondary_model = "buffalo_m"
        settings.quality_threshold = 0.75
        return settings

    @pytest.fixture
    def sample_image(self):
        """Create sample test image"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @patch('app.services.face_detection.dual_embedder.FaceAnalysis')
    def test_initialization(self, mock_face_analysis, mock_settings):
        """Test DualFaceEmbedder initialization"""
        with patch('app.core.config.settings', mock_settings):
            embedder = DualFaceEmbedder()

            # Should initialize successfully
            embedder.initialize()

            # Should have created embedders
            assert embedder.primary_embedder is not None
            assert embedder.secondary_embedder is not None

    @patch('app.services.face_detection.dual_embedder.FaceAnalysis')
    def test_extract_with_mock_faces(self, mock_face_analysis, mock_settings, sample_image):
        """Test face extraction with mocked face detection"""
        with patch('app.core.config.settings', mock_settings):
            embedder = DualFaceEmbedder()
            embedder.initialize()

            # Mock face detection results
            mock_primary_face = {
                'embedding': np.random.randn(512).astype(np.float32),
                'bbox': (100, 100, 200, 200),
                'det_score': 0.95,
                'quality': 0.9
            }

            mock_secondary_face = {
                'embedding': np.random.randn(512).astype(np.float32),
                'bbox': (105, 105, 205, 205),  # Slightly overlapping
                'det_score': 0.92,
                'quality': 0.88
            }

            # Mock the embedder methods
            embedder.primary_embedder.extract.return_value = [mock_primary_face]
            embedder.secondary_embedder.extract.return_value = [mock_secondary_face]

            # Extract faces
            faces = embedder.extract(sample_image)

            # Should return faces (merged and validated)
            assert len(faces) >= 1

            # Check that faces have expected attributes
            face = faces[0]
            assert 'embedding' in face
            assert 'bbox' in face
            assert 'source' in face  # Should be set by dual embedder

    def test_calculate_iou(self, mock_settings):
        """Test IoU calculation"""
        with patch('app.core.config.settings', mock_settings):
            embedder = DualFaceEmbedder()

            # Test identical boxes
            iou = embedder._calculate_iou((0, 0, 10, 10), (0, 0, 10, 10))
            assert iou == 1.0

            # Test no overlap
            iou = embedder._calculate_iou((0, 0, 10, 10), (20, 20, 30, 30))
            assert iou == 0.0

            # Test partial overlap
            iou = embedder._calculate_iou((0, 0, 20, 20), (10, 10, 30, 30))
            assert 0 < iou < 1

    def test_merge_face_detections(self, mock_settings):
        """Test merging face detections from two models"""
        with patch('app.core.config.settings', mock_settings):
            embedder = DualFaceEmbedder()

            # Create test faces - one overlapping pair
            primary_faces = [
                {
                    'embedding': np.random.randn(512).astype(np.float32),
                    'bbox': (100, 100, 200, 200),
                    'det_score': 0.9,
                    'quality': 0.85
                }
            ]

            secondary_faces = [
                {
                    'embedding': np.random.randn(512).astype(np.float32),
                    'bbox': (110, 110, 210, 210),  # Overlapping
                    'det_score': 0.88,
                    'quality': 0.82
                }
            ]

            sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            merged_faces = embedder._merge_face_detections(primary_faces, secondary_faces, sample_image)

            # Should merge overlapping faces
            assert len(merged_faces) >= 1

            # Check merged face has expected attributes
            merged_face = merged_faces[0]
            assert 'cross_validated' in merged_face
            assert merged_face['cross_validated'] is True

    def test_stats_tracking(self, mock_settings):
        """Test statistics tracking"""
        with patch('app.core.config.settings', mock_settings):
            embedder = DualFaceEmbedder()

            # Check initial stats
            assert embedder.stats['total_images'] == 0

            # Simulate processing
            embedder.stats['total_images'] = 5
            embedder.stats['faces_both'] = 3

            stats = embedder.get_stats()
            assert stats['total_images'] == 5
            assert stats['faces_both'] == 3
