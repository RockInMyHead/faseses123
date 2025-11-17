"""
Unit tests for DualQualityValidator
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.face_detection.dual_quality_validator import (
    DualQualityValidator,
    QualityMethod,
    QualityMetrics,
    ValidationResult
)


class TestDualQualityValidator:
    """Test DualQualityValidator functionality"""

    @pytest.fixture
    def sample_face(self):
        """Create sample face data"""
        return {
            'bbox': [100, 100, 200, 200],
            'det_score': 0.95,
            'quality': 0.9,
            'pose': [0.1, 0.05, 0.02],  # yaw, pitch, roll
            'landmarks': [[110, 120], [130, 125], [150, 140]]  # Sample landmarks
        }

    @pytest.fixture
    def sample_image(self):
        """Create sample test image"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def quality_validator(self):
        """Create quality validator instance"""
        return DualQualityValidator(
            methods=[QualityMethod.BASIC, QualityMethod.ADVANCED],
            adaptive_thresholds=True
        )

    def test_initialization(self, quality_validator):
        """Test DualQualityValidator initialization"""
        assert quality_validator.methods == [QualityMethod.BASIC, QualityMethod.ADVANCED]
        assert quality_validator.adaptive_thresholds is True
        assert len(quality_validator.method_performance) > 0

    def test_validate_face_dual_basic(self, quality_validator, sample_face, sample_image):
        """Test basic face validation"""
        result = quality_validator.validate_face_dual(sample_face, sample_image)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.final_score, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.quality_metrics, QualityMetrics)
        assert isinstance(result.recommendations, list)

        # Check score ranges
        assert 0.0 <= result.final_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_validate_face_dual_with_context(self, quality_validator, sample_face, sample_image):
        """Test validation with context information"""
        context = {
            'cluster_size': 5,
            'cluster_quality': 0.85,
            'image_quality': 0.9
        }

        result = quality_validator.validate_face_dual(sample_face, sample_image, context)

        assert isinstance(result, ValidationResult)
        assert result.validation_details['context_used'] is True

    def test_different_validation_methods(self, sample_face, sample_image):
        """Test different validation methods"""
        # Test with only basic method
        basic_validator = DualQualityValidator(methods=[QualityMethod.BASIC])
        basic_result = basic_validator.validate_face_dual(sample_face, sample_image)

        # Test with advanced methods
        advanced_validator = DualQualityValidator(methods=[QualityMethod.ADVANCED])
        advanced_result = advanced_validator.validate_face_dual(sample_face, sample_image)

        # Both should return valid results
        assert basic_result.final_score > 0
        assert advanced_result.final_score > 0

    def test_ensemble_validation(self, sample_face, sample_image):
        """Test ensemble validation method"""
        ensemble_validator = DualQualityValidator(methods=[QualityMethod.ENSEMBLE])
        result = ensemble_validator.validate_face_dual(sample_face, sample_image)

        assert result.quality_metrics.method == "ensemble"
        assert result.final_score > 0

    def test_quality_metrics_structure(self, quality_validator, sample_face, sample_image):
        """Test quality metrics structure"""
        result = quality_validator.validate_face_dual(sample_face, sample_image)

        metrics = result.quality_metrics

        # Check that all expected metrics are present
        required_attrs = [
            'detection_score', 'face_size_score', 'blur_score', 'brightness_score',
            'overall_score', 'processing_time'
        ]

        for attr in required_attrs:
            assert hasattr(metrics, attr)

        # Check score ranges
        assert 0.0 <= metrics.overall_score <= 1.0
        assert metrics.processing_time >= 0

    def test_adaptive_thresholds(self, quality_validator, sample_face, sample_image):
        """Test adaptive threshold adjustments"""
        # Run multiple validations to build history
        for _ in range(5):
            quality_validator.validate_face_dual(sample_face, sample_image)

        # Check that history is being collected
        assert len(quality_validator.quality_history) >= 5

        # Check statistics
        stats = quality_validator.get_statistics()
        assert 'total_validations' in stats
        assert 'average_quality' in stats
        assert stats['total_validations'] >= 5

    def test_recommendations_generation(self, quality_validator, sample_face, sample_image):
        """Test recommendations generation"""
        result = quality_validator.validate_face_dual(sample_face, sample_image)

        # Should always return a list of recommendations
        assert isinstance(result.recommendations, list)

        # For a good face, recommendations should be minimal
        if result.final_score > 0.8:
            assert len(result.recommendations) == 0

    def test_error_handling(self, quality_validator, sample_image):
        """Test error handling for invalid face data"""
        invalid_face = {'bbox': []}  # Missing required data

        result = quality_validator.validate_face_dual(invalid_face, sample_image)

        # Should still return a valid result (fallback metrics)
        assert isinstance(result, ValidationResult)
        assert result.final_score >= 0.0

    def test_symmetry_calculation(self, quality_validator):
        """Test symmetry score calculation"""
        # Create a symmetric face region
        symmetric_face = np.zeros((100, 100, 3), dtype=np.uint8)
        # Make left and right sides similar
        symmetric_face[:, :50] = 128
        symmetric_face[:, 50:] = 128

        score = quality_validator._calculate_symmetry_score(symmetric_face)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_blur_calculation(self, quality_validator):
        """Test blur score calculation"""
        # Create a sharp image
        sharp_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = [0, 0, 100, 100]

        score = quality_validator._calculate_blur_score(sharp_image, bbox)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_brightness_calculation(self, quality_validator):
        """Test brightness score calculation"""
        # Create a well-lit image
        bright_image = np.full((100, 100, 3), 150, dtype=np.uint8)
        bbox = [0, 0, 100, 100]

        score = quality_validator._calculate_brightness_score(bright_image, bbox)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        # Test optimal brightness
        assert score > 0.8  # Should be close to optimal

    def test_contrast_calculation(self, quality_validator):
        """Test contrast score calculation"""
        # Create high contrast image
        contrast_image = np.zeros((100, 100, 3), dtype=np.uint8)
        contrast_image[:50] = 255  # Bright top
        contrast_image[50:] = 0    # Dark bottom

        score = quality_validator._calculate_contrast_score(contrast_image)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_edge_sharpness_calculation(self, quality_validator):
        """Test edge sharpness calculation"""
        # Create image with clear edges
        edge_image = np.zeros((100, 100, 3), dtype=np.uint8)
        edge_image[:, 50:] = 255  # Vertical edge in middle

        score = quality_validator._calculate_edge_sharpness(edge_image)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_statistics_tracking(self, quality_validator, sample_face, sample_image):
        """Test statistics tracking"""
        # Run several validations
        for _ in range(3):
            quality_validator.validate_face_dual(sample_face, sample_image)

        stats = quality_validator.get_statistics()

        # Check statistics structure
        assert 'total_validations' in stats
        assert 'average_quality' in stats
        assert 'method_performance' in stats

        # Check method performance tracking
        method_perf = stats['method_performance']
        assert len(method_perf) > 0

        for method_stats in method_perf.values():
            assert 'count' in method_stats
            assert 'average' in method_stats
            assert 'std' in method_stats

    @patch('cv2.Laplacian')
    def test_opencv_fallback(self, mock_laplacian, quality_validator, sample_face, sample_image):
        """Test OpenCV fallback when cv2 is not available"""
        # Mock cv2 as unavailable
        with patch('app.services.face_detection.dual_quality_validator.cv2', None):
            result = quality_validator.validate_face_dual(sample_face, sample_image)

            # Should still work with fallback
            assert isinstance(result, ValidationResult)
            assert result.final_score >= 0.0
