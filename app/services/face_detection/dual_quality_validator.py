"""
Advanced Dual Quality Validator - продвинутая двойная валидация качества лиц

Реализует множественные методы оценки качества с ансамблевыми техниками
для максимальной точности определения качества детекции лиц.
"""
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ...core.logging import get_logger
from ...core.config import settings

logger = get_logger(__name__)


class QualityMethod(Enum):
    """Методы оценки качества"""
    BASIC = "basic"           # Базовая оценка
    ADVANCED = "advanced"     # Продвинутая оценка
    CNN_BASED = "cnn_based"   # CNN-based оценка
    ENSEMBLE = "ensemble"     # Ансамбль методов


@dataclass
class QualityMetrics:
    """Комплексные метрики качества лица"""
    # Основные метрики
    detection_score: float = 0.0
    face_size_score: float = 0.0
    blur_score: float = 0.0
    brightness_score: float = 0.0
    pose_score: float = 0.0
    symmetry_score: float = 0.0
    contrast_score: float = 0.0

    # Продвинутые метрики
    edge_sharpness: float = 0.0
    texture_richness: float = 0.0
    illumination_uniformity: float = 0.0
    facial_landmarks_confidence: float = 0.0

    # Композитные скоры
    overall_score: float = 0.0
    confidence_level: float = 0.0

    # Метаданные
    method: str = "unknown"
    processing_time: float = 0.0


@dataclass
class ValidationResult:
    """Результат двойной валидации"""
    final_score: float
    confidence: float
    quality_metrics: QualityMetrics
    validation_details: Dict[str, Any]
    recommendations: List[str]


class DualQualityValidator:
    """
    Продвинутая двойная валидация качества лиц

    Использует множественные методы оценки:
    1. Basic validation (размер, размытие, яркость)
    2. Advanced validation (симметрия, текстура, освещение)
    3. CNN-based validation (если доступно)
    4. Ensemble validation (ансамбль всех методов)
    """

    def __init__(self,
                 methods: List[QualityMethod] = None,
                 enable_cnn_validation: bool = False,
                 adaptive_thresholds: bool = True):
        """
        Args:
            methods: Список методов валидации для использования
            enable_cnn_validation: Включить CNN-based валидацию
            adaptive_thresholds: Адаптивные thresholds на основе данных
        """
        self.methods = methods or [QualityMethod.BASIC, QualityMethod.ADVANCED, QualityMethod.ENSEMBLE]
        self.enable_cnn_validation = enable_cnn_validation
        self.adaptive_thresholds = adaptive_thresholds

        # Статистика для адаптивных thresholds
        self.quality_history: List[float] = []
        self.method_performance: Dict[str, List[float]] = {
            method.value: [] for method in QualityMethod
        }

        # Weights для ensemble
        self.ensemble_weights = {
            QualityMethod.BASIC.value: 0.4,
            QualityMethod.ADVANCED.value: 0.4,
            QualityMethod.CNN_BASED.value: 0.2,
        }

        logger.info(f"Initialized DualQualityValidator with methods: {[m.value for m in self.methods]}")

    def validate_face_dual(self,
                          face: Dict,
                          img: np.ndarray,
                          context: Optional[Dict] = None) -> ValidationResult:
        """
        Двойная валидация качества лица

        Args:
            face: Данные о детектированном лице
            img: Исходное изображение
            context: Дополнительный контекст (кластер, позиция и т.д.)

        Returns:
            ValidationResult с комплексной оценкой
        """
        start_time = time.time()

        # Собираем результаты всех методов
        method_results = {}
        quality_metrics = {}

        for method in self.methods:
            try:
                if method == QualityMethod.BASIC:
                    result = self._validate_basic(face, img)
                elif method == QualityMethod.ADVANCED:
                    result = self._validate_advanced(face, img)
                elif method == QualityMethod.CNN_BASED:
                    result = self._validate_cnn_based(face, img)
                elif method == QualityMethod.ENSEMBLE:
                    result = self._validate_ensemble(face, img, method_results)
                else:
                    continue

                method_results[method.value] = result
                quality_metrics[method.value] = result

                # Сохраняем статистику
                self.method_performance[method.value].append(result.overall_score)

            except Exception as e:
                logger.warning(f"Failed to validate with method {method.value}: {e}")
                method_results[method.value] = self._create_fallback_metrics()

        # Вычисляем финальный результат
        final_result = self._compute_final_result(method_results, context)

        # Обновляем статистику для адаптивных thresholds
        self.quality_history.append(final_result.final_score)

        processing_time = time.time() - start_time
        final_result.quality_metrics.processing_time = processing_time

        return final_result

    def _validate_basic(self, face: Dict, img: np.ndarray) -> QualityMetrics:
        """Базовая валидация качества"""
        metrics = QualityMetrics(method="basic")
        start_time = time.time()

        # 1. Detection score
        metrics.detection_score = face.get('det_score', 0.5)

        # 2. Face size
        bbox = face.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            face_width = abs(x2 - x1)
            face_height = abs(y2 - y1)
            face_size = min(face_width, face_height)
            img_height, img_width = img.shape[:2]
            relative_size = face_size / min(img_width, img_height)

            # Оптимальный размер: 0.1-0.3 от размера изображения
            if 0.1 <= relative_size <= 0.3:
                metrics.face_size_score = 1.0
            elif relative_size < 0.1:
                metrics.face_size_score = relative_size / 0.1
            else:
                metrics.face_size_score = max(0.5, 1.0 - (relative_size - 0.3) / 0.2)

        # 3. Blur score
        metrics.blur_score = self._calculate_blur_score(img, bbox)

        # 4. Brightness score
        metrics.brightness_score = self._calculate_brightness_score(img, bbox)

        # 5. Pose score (если доступно)
        if 'pose' in face:
            pose = face['pose']
            if len(pose) >= 3:
                yaw, pitch, roll = pose[:3]
                max_angle = max(abs(yaw), abs(pitch), abs(roll))
                metrics.pose_score = max(0, 1.0 - max_angle / 45.0)  # 45 градусов - предел

        # Композитный скор
        weights = {
            'detection_score': 0.3,
            'face_size_score': 0.25,
            'blur_score': 0.25,
            'brightness_score': 0.15,
            'pose_score': 0.05
        }

        metrics.overall_score = sum(
            getattr(metrics, attr) * weight
            for attr, weight in weights.items()
        )

        metrics.processing_time = time.time() - start_time
        return metrics

    def _validate_advanced(self, face: Dict, img: np.ndarray) -> QualityMetrics:
        """Продвинутая валидация качества"""
        metrics = QualityMetrics(method="advanced")
        start_time = time.time()

        bbox = face.get('bbox', [])
        if len(bbox) != 4:
            return metrics

        x1, y1, x2, y2 = map(int, bbox)
        face_region = img[y1:y2, x1:x2]

        if face_region.size == 0:
            return metrics

        # 1. Symmetry score
        metrics.symmetry_score = self._calculate_symmetry_score(face_region)

        # 2. Contrast score
        metrics.contrast_score = self._calculate_contrast_score(face_region)

        # 3. Edge sharpness
        metrics.edge_sharpness = self._calculate_edge_sharpness(face_region)

        # 4. Texture richness
        metrics.texture_richness = self._calculate_texture_richness(face_region)

        # 5. Illumination uniformity
        metrics.illumination_uniformity = self._calculate_illumination_uniformity(face_region)

        # 6. Facial landmarks confidence (если доступны)
        if 'landmarks' in face:
            landmarks = face['landmarks']
            metrics.facial_landmarks_confidence = self._assess_landmarks_confidence(landmarks)

        # Композитный скор для продвинутых метрик
        weights = {
            'symmetry_score': 0.2,
            'contrast_score': 0.2,
            'edge_sharpness': 0.2,
            'texture_richness': 0.15,
            'illumination_uniformity': 0.15,
            'facial_landmarks_confidence': 0.1
        }

        advanced_score = sum(
            getattr(metrics, attr) * weight
            for attr, weight in weights.items()
            if getattr(metrics, attr) > 0
        )

        # Комбинируем с базовыми метриками
        basic_metrics = self._validate_basic(face, img)
        metrics.overall_score = (basic_metrics.overall_score * 0.6) + (advanced_score * 0.4)

        # Копируем базовые метрики
        for attr in ['detection_score', 'face_size_score', 'blur_score', 'brightness_score', 'pose_score']:
            setattr(metrics, attr, getattr(basic_metrics, attr))

        metrics.processing_time = time.time() - start_time
        return metrics

    def _validate_cnn_based(self, face: Dict, img: np.ndarray) -> QualityMetrics:
        """CNN-based валидация качества (заглушка для будущей реализации)"""
        metrics = QualityMetrics(method="cnn_based")

        # Пока возвращаем базовую оценку
        # В будущем здесь будет CNN модель для оценки качества
        basic_metrics = self._validate_basic(face, img)
        metrics.overall_score = basic_metrics.overall_score

        # Имитация дополнительных метрик
        metrics.confidence_level = 0.8  # CNN уверенность

        return metrics

    def _validate_ensemble(self, face: Dict, img: np.ndarray,
                          method_results: Dict[str, QualityMetrics]) -> QualityMetrics:
        """Ансамблевая валидация"""
        metrics = QualityMetrics(method="ensemble")

        if not method_results:
            return metrics

        # Взвешенное среднее результатов разных методов
        total_weight = 0
        weighted_sum = 0

        for method_name, method_metrics in method_results.items():
            if method_name == QualityMethod.ENSEMBLE.value:
                continue

            weight = self.ensemble_weights.get(method_name, 0.33)
            weighted_sum += method_metrics.overall_score * weight
            total_weight += weight

        if total_weight > 0:
            metrics.overall_score = weighted_sum / total_weight
        else:
            # Fallback к среднему
            scores = [m.overall_score for m in method_results.values()
                     if m.method != QualityMethod.ENSEMBLE.value]
            metrics.overall_score = np.mean(scores) if scores else 0.5

        # Confidence based on agreement between methods
        scores = [m.overall_score for m in method_results.values()
                 if m.method != QualityMethod.ENSEMBLE.value]
        if len(scores) > 1:
            std_dev = np.std(scores)
            metrics.confidence_level = max(0, 1.0 - std_dev)  # Меньше разброс - выше уверенность
        else:
            metrics.confidence_level = 0.5

        return metrics

    def _compute_final_result(self,
                            method_results: Dict[str, QualityMetrics],
                            context: Optional[Dict] = None) -> ValidationResult:
        """Вычисление финального результата валидации"""

        # Используем ensemble как основной результат, если доступен
        if QualityMethod.ENSEMBLE.value in method_results:
            primary_metrics = method_results[QualityMethod.ENSEMBLE.value]
        elif QualityMethod.ADVANCED.value in method_results:
            primary_metrics = method_results[QualityMethod.ADVANCED.value]
        else:
            primary_metrics = list(method_results.values())[0] if method_results else QualityMetrics()

        # Адаптивная корректировка на основе контекста
        adjusted_score = self._apply_context_adjustments(primary_metrics.overall_score, context)

        # Вычисление confidence
        confidence = self._calculate_overall_confidence(method_results, context)

        # Генерация рекомендаций
        recommendations = self._generate_recommendations(primary_metrics, method_results)

        return ValidationResult(
            final_score=adjusted_score,
            confidence=confidence,
            quality_metrics=primary_metrics,
            validation_details={
                'method_results': {k: v.__dict__ for k, v in method_results.items()},
                'context_used': context is not None,
                'adaptive_adjustments': adjusted_score != primary_metrics.overall_score
            },
            recommendations=recommendations
        )

    def _apply_context_adjustments(self, base_score: float, context: Optional[Dict]) -> float:
        """Адаптивная корректировка скора на основе контекста"""
        if not context or not self.adaptive_thresholds:
            return base_score

        adjusted_score = base_score

        # Корректировка на основе позиции в кластере
        if 'cluster_size' in context:
            cluster_size = context['cluster_size']
            if cluster_size == 1:
                # Для одиночных лиц более строгие требования
                adjusted_score *= 0.9
            elif cluster_size > 5:
                # Для больших кластеров можно быть мягче
                adjusted_score = min(1.0, adjusted_score * 1.05)

        # Корректировка на основе качества кластера
        if 'cluster_quality' in context:
            cluster_quality = context['cluster_quality']
            # Если кластер хорошего качества, немного повышаем скор
            quality_boost = (cluster_quality - 0.8) * 0.1  # max 0.1 boost
            adjusted_score = min(1.0, adjusted_score + max(0, quality_boost))

        return adjusted_score

    def _calculate_overall_confidence(self,
                                    method_results: Dict[str, QualityMetrics],
                                    context: Optional[Dict]) -> float:
        """Расчет общей уверенности в оценке"""

        confidences = []

        # Confidence от методов
        for metrics in method_results.values():
            if hasattr(metrics, 'confidence_level') and metrics.confidence_level > 0:
                confidences.append(metrics.confidence_level)
            else:
                # Эвристическая confidence на основе стабильности скора
                score_std = np.std([m.overall_score for m in method_results.values()])
                method_confidence = max(0, 1.0 - score_std * 2)
                confidences.append(method_confidence)

        # Confidence от контекста
        if context:
            context_confidence = 0.8  # Базовая confidence от контекста
            confidences.append(context_confidence)

        return np.mean(confidences) if confidences else 0.5

    def _generate_recommendations(self,
                                primary_metrics: QualityMetrics,
                                method_results: Dict[str, QualityMetrics]) -> List[str]:
        """Генерация рекомендаций по улучшению качества"""

        recommendations = []

        # Анализ основных проблем
        if primary_metrics.face_size_score < 0.7:
            recommendations.append("Лицо слишком маленькое - приблизьте камеру")

        if primary_metrics.blur_score < 0.6:
            recommendations.append("Изображение размытое - улучшите освещение или фокус")

        if primary_metrics.brightness_score < 0.6:
            recommendations.append("Плохое освещение - используйте равномерное освещение")

        if primary_metrics.symmetry_score < 0.7:
            recommendations.append("Лицо повернуто - смотрите прямо в камеру")

        # Анализ согласованности методов
        scores = [m.overall_score for m in method_results.values()]
        if len(scores) > 1 and np.std(scores) > 0.2:
            recommendations.append("Методы оценки расходятся - перепроверьте изображение")

        # Рекомендации по confidence
        if primary_metrics.confidence_level < 0.7:
            recommendations.append("Низкая уверенность оценки - рассмотрите повторную съемку")

        return recommendations

    # Методы оценки конкретных характеристик
    def _calculate_blur_score(self, img: np.ndarray, bbox: List[int]) -> float:
        """Оценка размытия лица"""
        try:
            import cv2
        except ImportError:
            return 0.5

        if len(bbox) != 4:
            return 0.5

        x1, y1, x2, y2 = bbox
        face_region = img[y1:y2, x1:x2]

        if face_region.size == 0:
            return 0.5

        # Преобразование в grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region

        # Вычисление Laplacian variance (мера резкости)
        laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
        variance = laplacian.var()

        # Нормализация
        blur_score = min(variance / 500.0, 1.0)
        return blur_score

    def _calculate_brightness_score(self, img: np.ndarray, bbox: List[int]) -> float:
        """Оценка яркости лица"""
        if len(bbox) != 4:
            return 0.5

        x1, y1, x2, y2 = bbox
        face_region = img[y1:y2, x1:x2]

        if face_region.size == 0:
            return 0.5

        # Преобразование в grayscale
        if len(face_region.shape) == 3:
            gray = np.dot(face_region[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = face_region

        brightness = np.mean(gray) / 255.0

        # Оптимальная яркость 0.3-0.7
        if 0.3 <= brightness <= 0.7:
            return 1.0
        elif brightness < 0.3:
            return brightness / 0.3
        else:
            return (1.0 - brightness) / 0.3

    def _calculate_symmetry_score(self, face_region: np.ndarray) -> float:
        """Оценка симметрии лица"""
        try:
            # Разделение лица на левую и правую половины
            h, w = face_region.shape[:2]
            mid = w // 2

            left_half = face_region[:, :mid]
            right_half = face_region[:, mid:]

            # Переворот правой половины
            right_flipped = np.fliplr(right_half)

            # Изменение размера для совпадения
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_flipped = right_flipped[:, :min_width]

            # Вычисление корреляции
            if left_half.size > 0 and right_flipped.size > 0:
                correlation = np.corrcoef(left_half.flatten(), right_flipped.flatten())[0, 1]
                symmetry_score = max(0, correlation)  # Корреляция от -1 до 1, берем положительную часть
                return symmetry_score
            else:
                return 0.5

        except Exception:
            return 0.5

    def _calculate_contrast_score(self, face_region: np.ndarray) -> float:
        """Оценка контрастности лица"""
        try:
            # Преобразование в grayscale
            if len(face_region.shape) == 3:
                gray = np.dot(face_region[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = face_region

            # RMS contrast
            contrast = np.sqrt(np.mean((gray - np.mean(gray))**2)) / np.mean(gray)

            # Нормализация (оптимальный контраст ~0.1-0.3)
            if 0.1 <= contrast <= 0.3:
                return 1.0
            elif contrast < 0.1:
                return contrast / 0.1
            else:
                return max(0.5, 1.0 - (contrast - 0.3) / 0.2)

        except Exception:
            return 0.5

    def _calculate_edge_sharpness(self, face_region: np.ndarray) -> float:
        """Оценка резкости краев"""
        try:
            import cv2
        except ImportError:
            return 0.5

        try:
            # Преобразование в grayscale
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_region

            # Canny edge detection
            edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            # Нормализация (оптимальная плотность краев ~0.1-0.3)
            if 0.1 <= edge_density <= 0.3:
                return 1.0
            elif edge_density < 0.1:
                return edge_density / 0.1
            else:
                return max(0.5, 1.0 - (edge_density - 0.3) / 0.2)

        except Exception:
            return 0.5

    def _calculate_texture_richness(self, face_region: np.ndarray) -> float:
        """Оценка богатства текстуры"""
        try:
            # Преобразование в grayscale
            if len(face_region.shape) == 3:
                gray = np.dot(face_region[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = face_region

            # Вычисление стандартного отклонения как меры вариативности текстуры
            texture_std = np.std(gray)

            # Нормализация (оптимальное std ~20-60)
            if 20 <= texture_std <= 60:
                return 1.0
            elif texture_std < 20:
                return texture_std / 20
            else:
                return max(0.5, 1.0 - (texture_std - 60) / 40)

        except Exception:
            return 0.5

    def _calculate_illumination_uniformity(self, face_region: np.ndarray) -> float:
        """Оценка равномерности освещения"""
        try:
            # Преобразование в grayscale
            if len(face_region.shape) == 3:
                gray = np.dot(face_region[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = face_region

            # Разделение на блоки и анализ вариативности
            h, w = gray.shape
            block_size = min(h, w) // 4

            if block_size < 10:
                return 0.5

            blocks = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    blocks.append(np.mean(block))

            if len(blocks) < 4:
                return 0.5

            # Коэффициент вариации между блоками
            blocks_mean = np.mean(blocks)
            blocks_std = np.std(blocks)

            if blocks_mean > 0:
                cv = blocks_std / blocks_mean
                # Низкий CV означает равномерное освещение
                uniformity_score = max(0, 1.0 - cv * 2)
                return uniformity_score
            else:
                return 0.5

        except Exception:
            return 0.5

    def _assess_landmarks_confidence(self, landmarks: List) -> float:
        """Оценка уверенности facial landmarks"""
        if not landmarks:
            return 0.0

        # Простая эвристика: чем больше landmarks, тем лучше
        num_landmarks = len(landmarks)

        if num_landmarks >= 68:  # Полный набор landmarks
            return 1.0
        elif num_landmarks >= 5:  # Основные точки
            return 0.8
        else:
            return num_landmarks / 5.0

    def _create_fallback_metrics(self) -> QualityMetrics:
        """Создание fallback метрик при ошибке"""
        metrics = QualityMetrics(method="fallback")
        metrics.overall_score = 0.5
        metrics.confidence_level = 0.3
        return metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики работы валидатора"""
        return {
            'total_validations': len(self.quality_history),
            'average_quality': np.mean(self.quality_history) if self.quality_history else 0.0,
            'quality_std': np.std(self.quality_history) if self.quality_history else 0.0,
            'method_performance': {
                method: {
                    'count': len(scores),
                    'average': np.mean(scores) if scores else 0.0,
                    'std': np.std(scores) if scores else 0.0
                }
                for method, scores in self.method_performance.items()
            }
        }
