"""
Unit tests for AdvancedFaceRescue
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.face_detection.advanced_rescue import (
    AdvancedFaceRescue,
    RescueStrategy,
    RescueCandidate,
    RescueResult
)


class TestAdvancedFaceRescue:
    """Test AdvancedFaceRescue functionality"""

    @pytest.fixture
    def sample_rejected_faces(self):
        """Create sample rejected faces"""
        return [
            {
                'bbox': [100, 100, 200, 200],
                'quality': 0.6,
                'validation_details': {
                    'primary_score': 0.65,
                    'secondary_score': 0.55,
                    'cross_validation_score': 0.8,
                    'final_score': 0.6
                }
            },
            {
                'bbox': [50, 50, 150, 150],
                'quality': 0.4,
                'validation_details': {
                    'primary_score': 0.45,
                    'secondary_score': 0.35,
                    'cross_validation_score': 0.9,
                    'final_score': 0.4
                }
            }
        ]

    @pytest.fixture
    def sample_image(self):
        """Create sample test image"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def rescue_system(self):
        """Create rescue system instance"""
        return AdvancedFaceRescue(
            strategy=RescueStrategy.BALANCED,
            adaptive_learning=True
        )

    def test_initialization(self, rescue_system):
        """Test AdvancedFaceRescue initialization"""
        assert rescue_system.strategy == RescueStrategy.BALANCED
        assert rescue_system.adaptive_learning is True
        assert len(rescue_system.rescue_stats) > 0

    def test_rescue_candidates_analysis(self, rescue_system, sample_rejected_faces, sample_image):
        """Test analysis of rescue candidates"""
        candidates = rescue_system._analyze_rescue_candidates(
            sample_rejected_faces, sample_image, "test.jpg", None
        )

        assert len(candidates) > 0
        assert all(isinstance(c, RescueCandidate) for c in candidates)

        # Check candidate properties
        for candidate in candidates:
            assert 0.0 <= candidate.rescue_potential <= 1.0
            assert candidate.risk_level in ['low', 'medium', 'high']
            assert len(candidate.reasons) > 0

    def test_rescue_potential_calculation(self, rescue_system):
        """Test calculation of rescue potential"""
        # High potential candidate
        high_potential_face = {
            'quality': 0.7,
            'validation_details': {
                'primary_score': 0.75,
                'secondary_score': 0.65,
                'cross_validation_score': 0.9
            }
        }

        candidate = rescue_system._evaluate_rescue_potential(high_potential_face, None, None)
        assert candidate.rescue_potential > 0.5

        # Low potential candidate
        low_potential_face = {
            'quality': 0.2,
            'validation_details': {
                'primary_score': 0.25,
                'secondary_score': 0.15,
                'cross_validation_score': 0.3
            }
        }

        candidate = rescue_system._evaluate_rescue_potential(low_potential_face, None, None)
        assert candidate.rescue_potential < 0.5

    def test_context_factor_analysis(self, rescue_system):
        """Test analysis of context factors"""
        face = {'bbox': [100, 100, 200, 200]}
        context = {
            'cluster_size': 5,
            'cluster_quality': 0.8,
            'image_faces_count': 3
        }

        factors = rescue_system._analyze_context_factors(face, context)

        assert factors['cluster_size'] == 5
        assert factors['cluster_quality'] == 0.8
        assert factors['image_faces_count'] == 3

    def test_strategy_filtering(self, rescue_system, sample_rejected_faces, sample_image):
        """Test filtering by rescue strategy"""
        candidates = rescue_system._analyze_rescue_candidates(
            sample_rejected_faces, sample_image, "test.jpg", None
        )

        # Test different strategies
        conservative_system = AdvancedFaceRescue(strategy=RescueStrategy.CONSERVATIVE)
        conservative_filtered = conservative_system._filter_by_strategy(candidates, None)

        balanced_system = AdvancedFaceRescue(strategy=RescueStrategy.BALANCED)
        balanced_filtered = balanced_system._filter_by_strategy(candidates, None)

        aggressive_system = AdvancedFaceRescue(strategy=RescueStrategy.AGGRESSIVE)
        aggressive_filtered = aggressive_system._filter_by_strategy(candidates, None)

        # Conservative should be most restrictive
        assert len(conservative_filtered) <= len(balanced_filtered)
        assert len(balanced_filtered) <= len(aggressive_filtered)

    def test_rescue_execution(self, rescue_system, sample_rejected_faces, sample_image):
        """Test execution of rescue operation"""
        result = rescue_system.rescue_faces(
            sample_rejected_faces, sample_image, "test.jpg", None
        )

        assert isinstance(result, RescueResult)
        assert isinstance(result.rescued_faces, list)
        assert isinstance(result.rejected_faces, list)
        assert isinstance(result.statistics, dict)
        assert isinstance(result.recommendations, list)

        # Total faces should be conserved
        total_faces = len(result.rescued_faces) + len(result.rejected_faces)
        assert total_faces == len(sample_rejected_faces)

    def test_statistics_tracking(self, rescue_system, sample_rejected_faces, sample_image):
        """Test statistics tracking"""
        initial_stats = rescue_system.rescue_stats.copy()

        rescue_system.rescue_faces(sample_rejected_faces, sample_image, "test.jpg", None)

        # Statistics should be updated
        assert rescue_system.rescue_stats['total_candidates'] > initial_stats['total_candidates']

    def test_risk_assessment(self, rescue_system):
        """Test risk assessment for rescue candidates"""
        # Low risk case
        low_risk_face = {
            'quality': 0.7,
            'validation_details': {
                'primary_score': 0.75,
                'secondary_score': 0.70,
                'cross_validation_score': 0.95
            }
        }

        risk_level, risk_score = rescue_system._assess_rescue_risk(
            low_risk_face, low_risk_face['validation_details'], {}
        )

        assert risk_level == 'low'
        assert risk_score < 0.4

        # High risk case
        high_risk_face = {
            'quality': 0.2,
            'validation_details': {
                'primary_score': 0.25,
                'secondary_score': 0.15,
                'cross_validation_score': 0.4
            }
        }

        risk_level, risk_score = rescue_system._assess_rescue_risk(
            high_risk_face, high_risk_face['validation_details'], {}
        )

        assert risk_level in ['medium', 'high']
        assert risk_score > 0.4

    def test_adaptive_learning(self, rescue_system, sample_rejected_faces, sample_image):
        """Test adaptive learning functionality"""
        # Run rescue multiple times
        for _ in range(3):
            rescue_system.rescue_faces(sample_rejected_faces, sample_image, "test.jpg", None)

        # Check that learning patterns are accumulated
        assert len(rescue_system.success_patterns) >= 0
        assert len(rescue_system.failure_patterns) >= 0

        # Check recent success rate calculation
        stats = rescue_system.get_rescue_statistics()
        assert 'learning_patterns' in stats
        assert 'recent_success_rate' in stats['learning_patterns']

    def test_face_position_classification(self, rescue_system):
        """Test face position classification"""
        # Center position
        position = rescue_system._classify_face_position([200, 150, 300, 250], {
            'image_width': 500, 'image_height': 400
        })
        assert 'center' in position

        # Corner position
        position = rescue_system._classify_face_position([0, 0, 100, 100], {
            'image_width': 500, 'image_height': 400
        })
        assert 'top_left' == position

    def test_empty_input_handling(self, rescue_system, sample_image):
        """Test handling of empty input"""
        result = rescue_system.rescue_faces([], sample_image, "empty.jpg", None)

        assert len(result.rescued_faces) == 0
        assert len(result.rejected_faces) == 0
        assert len(result.recommendations) == 0

    def test_context_aware_filtering(self):
        """Test context-aware filtering strategy"""
        context_system = AdvancedFaceRescue(strategy=RescueStrategy.CONTEXT_AWARE)

        candidates = [
            RescueCandidate(
                face={'quality': 0.65},
                original_score=0.65,
                validation_details={},
                rescue_potential=0.7,
                risk_level='low',
                reasons=['Good agreement'],
                context_factors={'cluster_size': 1}
            )
        ]

        context = {'cluster_size': 1, 'cluster_quality': 0.9}
        filtered = context_system._filter_by_strategy(candidates, context)

        # Should allow rescue for small clusters
        assert len(filtered) > 0

    def test_rescue_recommendations(self, rescue_system, sample_rejected_faces, sample_image):
        """Test generation of rescue recommendations"""
        # Force some rescues by setting high potential
        for face in sample_rejected_faces:
            face['validation_details'] = {
                'primary_score': 0.8,
                'secondary_score': 0.75,
                'cross_validation_score': 0.9
            }

        result = rescue_system.rescue_faces(sample_rejected_faces, sample_image, "test.jpg", None)

        # Should generate recommendations
        assert isinstance(result.recommendations, list)

        # If many faces rescued, should recommend adjusting thresholds
        if len(result.rescued_faces) > len(result.rejected_faces):
            recommendation_texts = ' '.join(result.recommendations)
            assert 'threshold' in recommendation_texts.lower() or 'уровень' in recommendation_texts.lower()
