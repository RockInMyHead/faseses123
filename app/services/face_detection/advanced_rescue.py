"""
Advanced Face Rescue - продвинутая система повторного распознавания сомнительных случаев

Реализует интеллектуальные механизмы спасения низкокачественных лиц,
которые могут быть полезны для кластеризации, но были отфильтрованы
стандартными критериями качества.
"""
import time
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ...core.logging import get_logger
from ...core.config import settings
from .dual_quality_validator import DualQualityValidator, ValidationResult, QualityMethod

logger = get_logger(__name__)


class RescueStrategy(Enum):
    """Стратегии rescue низкокачественных лиц"""
    CONSERVATIVE = "conservative"    # Строгие критерии, минимум false positives
    BALANCED = "balanced"           # Сбалансированный подход
    AGGRESSIVE = "aggressive"       # Максимальное спасение, допускает false positives
    CONTEXT_AWARE = "context_aware" # Учитывает контекст кластеризации


@dataclass
class RescueCandidate:
    """Кандидат на rescue"""
    face: Dict[str, Any]
    original_score: float
    validation_details: Dict[str, Any]
    rescue_potential: float  # 0-1, вероятность успешного rescue
    risk_level: str  # 'low', 'medium', 'high'
    reasons: List[str]  # Причины для rescue
    context_factors: Dict[str, Any]


@dataclass
class RescueResult:
    """Результат rescue операции"""
    rescued_faces: List[Dict[str, Any]]
    rejected_faces: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    recommendations: List[str]


class AdvancedFaceRescue:
    """
    Продвинутая система повторного распознавания сомнительных случаев

    Анализирует отклоненные лица и принимает решения о rescue на основе:
    - Качества валидации и согласованности методов
    - Контекста кластеризации
    - Статистики предыдущих rescue операций
    - Риска false positives
    """

    def __init__(self,
                 strategy: RescueStrategy = RescueStrategy.BALANCED,
                 quality_validator: Optional[DualQualityValidator] = None,
                 max_rescue_attempts: int = 3,
                 adaptive_learning: bool = True):
        """
        Args:
            strategy: Стратегия rescue
            quality_validator: Валидатор качества для повторной оценки
            max_rescue_attempts: Максимальное количество попыток rescue
            adaptive_learning: Адаптивное обучение на основе результатов
        """
        self.strategy = strategy
        self.quality_validator = quality_validator or DualQualityValidator()
        self.max_rescue_attempts = max_rescue_attempts
        self.adaptive_learning = adaptive_learning

        # Статистика rescue операций
        self.rescue_stats = {
            'total_candidates': 0,
            'rescued_faces': 0,
            'successful_rescues': 0,
            'failed_rescues': 0,
            'by_risk_level': {'low': 0, 'medium': 0, 'high': 0},
            'by_strategy': {s.value: 0 for s in RescueStrategy},
            'context_factors': {}
        }

        # Обучение на результатах
        self.success_patterns = []
        self.failure_patterns = []

        logger.info(f"Initialized AdvancedFaceRescue with strategy: {strategy.value}")

    def rescue_faces(self,
                    rejected_faces: List[Dict],
                    img: np.ndarray,
                    img_path: str,
                    context: Optional[Dict] = None) -> RescueResult:
        """
        Выполнить rescue операцию для отклоненных лиц

        Args:
            rejected_faces: Список отклоненных лиц
            img: Исходное изображение
            img_path: Путь к изображению
            context: Контекст кластеризации

        Returns:
            RescueResult с rescued и rejected лицами
        """
        start_time = time.time()

        if not rejected_faces:
            return RescueResult([], [], {}, [])

        # Анализ кандидатов на rescue
        candidates = self._analyze_rescue_candidates(rejected_faces, img, img_path, context)

        # Фильтрация по стратегии
        viable_candidates = self._filter_by_strategy(candidates, context)

        # Попытка rescue
        rescued_faces = []
        rejected_faces_final = []

        for candidate in viable_candidates:
            if self._attempt_rescue(candidate, img, context):
                rescued_faces.append(candidate.face)
                self.rescue_stats['rescued_faces'] += 1
                self.rescue_stats['by_risk_level'][candidate.risk_level] += 1

                # Маркировка rescued лица
                candidate.face['rescued'] = True
                candidate.face['rescue_info'] = {
                    'original_score': candidate.original_score,
                    'rescue_strategy': self.strategy.value,
                    'risk_level': candidate.risk_level,
                    'reasons': candidate.reasons,
                    'context_factors': candidate.context_factors
                }

                logger.debug(f"Rescued face in {img_path}: score {candidate.original_score:.3f} "
                           f"(risk: {candidate.risk_level})")
            else:
                rejected_faces_final.append(candidate.face)

        # Добавляем не-viable кандидатов к финальным rejected
        rejected_faces_final.extend([c.face for c in candidates if c not in viable_candidates])

        # Обновление статистики
        processing_time = time.time() - start_time
        self.rescue_stats['total_candidates'] += len(candidates)

        # Генерация рекомендаций
        recommendations = self._generate_rescue_recommendations(rescued_faces, rejected_faces_final, context)

        # Адаптивное обучение
        if self.adaptive_learning:
            self._update_learning_model(rescued_faces, rejected_faces_final)

        result = RescueResult(
            rescued_faces=rescued_faces,
            rejected_faces=rejected_faces_final,
            statistics=self._compile_statistics(processing_time),
            recommendations=recommendations
        )

        logger.info(f"Rescue operation completed: {len(rescued_faces)} rescued, "
                   f"{len(rejected_faces_final)} rejected from {len(rejected_faces)} candidates")

        return result

    def _analyze_rescue_candidates(self,
                                 rejected_faces: List[Dict],
                                 img: np.ndarray,
                                 img_path: str,
                                 context: Optional[Dict]) -> List[RescueCandidate]:
        """
        Анализ кандидатов на rescue
        """
        candidates = []

        for face in rejected_faces:
            candidate = self._evaluate_rescue_potential(face, img, context)
            if candidate.rescue_potential > 0:
                candidates.append(candidate)

        # Сортировка по потенциалу rescue (лучшие сначала)
        candidates.sort(key=lambda c: c.rescue_potential, reverse=True)

        return candidates

    def _evaluate_rescue_potential(self,
                                 face: Dict,
                                 img: np.ndarray,
                                 context: Optional[Dict]) -> RescueCandidate:
        """
        Оценка потенциала rescue для конкретного лица
        """
        original_score = face.get('quality', 0.0)
        validation_details = face.get('validation_details', {})

        # Базовый потенциал на основе качества
        base_potential = self._calculate_base_rescue_potential(original_score, validation_details)

        # Корректировка на основе контекста
        context_factors = self._analyze_context_factors(face, context)
        context_multiplier = self._calculate_context_multiplier(context_factors)

        # Оценка риска
        risk_level, risk_score = self._assess_rescue_risk(face, validation_details, context_factors)

        # Финальный потенциал
        final_potential = min(base_potential * context_multiplier, 1.0)

        # Причины для rescue
        reasons = self._identify_rescue_reasons(face, validation_details, context_factors)

        return RescueCandidate(
            face=face,
            original_score=original_score,
            validation_details=validation_details,
            rescue_potential=final_potential,
            risk_level=risk_level,
            reasons=reasons,
            context_factors=context_factors
        )

    def _calculate_base_rescue_potential(self,
                                       original_score: float,
                                       validation_details: Dict) -> float:
        """
        Расчет базового потенциала rescue на основе качества
        """
        if not validation_details:
            # Fallback для лиц без детальной валидации
            if original_score >= 0.4:  # 40% от threshold
                return min(original_score * 2.5, 0.8)
            return 0.0

        primary_score = validation_details.get('primary_score', 0.0)
        secondary_score = validation_details.get('secondary_score', 0.0)
        agreement = validation_details.get('cross_validation_score', 0.0)

        # Потенциал = среднее качество * согласованность * корректировка
        avg_quality = (primary_score + secondary_score) / 2.0

        # Согласованность повышает уверенность
        agreement_bonus = agreement * 0.2

        # Расстояние до threshold влияет на потенциал
        threshold_distance = settings.quality_threshold - avg_quality
        if threshold_distance > 0:
            distance_factor = max(0, 1.0 - threshold_distance / settings.quality_threshold)
        else:
            distance_factor = 1.0  # Уже выше threshold

        base_potential = (avg_quality + agreement_bonus) * distance_factor
        return min(base_potential, 1.0)

    def _analyze_context_factors(self, face: Dict, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Анализ контекстных факторов, влияющих на rescue
        """
        factors = {
            'cluster_size': 1,
            'cluster_quality': 0.5,
            'image_faces_count': 1,
            'face_position': 'unknown',
            'similarity_to_cluster': 0.0,
            'temporal_context': 'none'
        }

        if not context:
            return factors

        # Размер кластера
        factors['cluster_size'] = context.get('cluster_size', 1)

        # Качество кластера
        factors['cluster_quality'] = context.get('cluster_quality', 0.5)

        # Количество лиц на изображении
        factors['image_faces_count'] = context.get('image_faces_count', 1)

        # Позиция лица
        if 'bbox' in face:
            bbox = face['bbox']
            # Определение позиции (центр, край и т.д.)
            factors['face_position'] = self._classify_face_position(bbox, context)

        # Сходство с кластером
        factors['similarity_to_cluster'] = context.get('cluster_similarity', 0.0)

        # Временной контекст
        factors['temporal_context'] = context.get('temporal_context', 'none')

        return factors

    def _calculate_context_multiplier(self, context_factors: Dict) -> float:
        """
        Расчет множителя на основе контекстных факторов
        """
        multiplier = 1.0

        # Модификатор размера кластера
        cluster_size = context_factors['cluster_size']
        if cluster_size == 1:
            # Для одиночных кластеров rescue менее желателен
            multiplier *= 0.7
        elif cluster_size <= 3:
            # Маленькие кластеры - умеренное повышение
            multiplier *= 1.2
        elif cluster_size >= 10:
            # Большие кластеры - можно быть более лояльным
            multiplier *= 1.3

        # Модификатор качества кластера
        cluster_quality = context_factors['cluster_quality']
        if cluster_quality > 0.8:
            # Хороший кластер - меньше риска от rescue
            multiplier *= 1.1
        elif cluster_quality < 0.6:
            # Плохой кластер - rescue может помочь
            multiplier *= 1.4

        # Модификатор количества лиц
        faces_count = context_factors['image_faces_count']
        if faces_count == 1:
            # Единственное лицо - rescue более важен
            multiplier *= 1.3
        elif faces_count > 5:
            # Много лиц - можно быть строже
            multiplier *= 0.9

        return multiplier

    def _assess_rescue_risk(self,
                           face: Dict,
                           validation_details: Dict,
                           context_factors: Dict) -> Tuple[str, float]:
        """
        Оценка риска rescue операции
        """
        risk_score = 0.0

        # Риск на основе качества
        if validation_details:
            agreement = validation_details.get('cross_validation_score', 0.5)
            avg_quality = (validation_details.get('primary_score', 0) +
                          validation_details.get('secondary_score', 0)) / 2.0

            # Низкая согласованность = высокий риск
            if agreement < 0.7:
                risk_score += 0.3

            # Низкое качество = высокий риск
            if avg_quality < 0.3:
                risk_score += 0.4
        else:
            # Нет детальной валидации = средний риск
            risk_score += 0.2

        # Риск на основе контекста
        cluster_size = context_factors['cluster_size']
        if cluster_size == 1:
            risk_score += 0.2  # Риск создания нового кластера

        cluster_quality = context_factors['cluster_quality']
        if cluster_quality < 0.5:
            risk_score -= 0.1  # Плохой кластер может выиграть от rescue

        # Определение уровня риска
        if risk_score < 0.3:
            risk_level = 'low'
        elif risk_score < 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        return risk_level, risk_score

    def _identify_rescue_reasons(self,
                               face: Dict,
                               validation_details: Dict,
                               context_factors: Dict) -> List[str]:
        """
        Идентификация причин для rescue
        """
        reasons = []

        # Причины качества
        if validation_details:
            primary = validation_details.get('primary_score', 0)
            secondary = validation_details.get('secondary_score', 0)
            agreement = validation_details.get('cross_validation_score', 0)

            if primary >= 0.5 or secondary >= 0.5:
                reasons.append("Одна модель оценила высоко")

            if agreement > 0.8:
                reasons.append("Высокая согласованность моделей")

        # Причины контекста
        if context_factors['cluster_size'] == 1:
            reasons.append("Единственный кандидат в кластер")

        if context_factors['cluster_quality'] < 0.6:
            reasons.append("Низкое качество кластера")

        if context_factors['image_faces_count'] == 1:
            reasons.append("Единственное лицо на изображении")

        if not reasons:
            reasons.append("Близко к порогу качества")

        return reasons

    def _filter_by_strategy(self,
                          candidates: List[RescueCandidate],
                          context: Optional[Dict]) -> List[RescueCandidate]:
        """
        Фильтрация кандидатов по выбранной стратегии
        """
        if self.strategy == RescueStrategy.CONSERVATIVE:
            # Только низкий риск, высокий потенциал
            return [c for c in candidates
                   if c.risk_level == 'low' and c.rescue_potential > 0.7]

        elif self.strategy == RescueStrategy.BALANCED:
            # Средний риск, хороший потенциал
            return [c for c in candidates
                   if c.risk_level in ['low', 'medium'] and c.rescue_potential > 0.5]

        elif self.strategy == RescueStrategy.AGGRESSIVE:
            # Высокий риск разрешен, минимальный потенциал
            return [c for c in candidates if c.rescue_potential > 0.3]

        elif self.strategy == RescueStrategy.CONTEXT_AWARE:
            # Учитывает контекст для принятия решения
            return self._context_aware_filtering(candidates, context)

        return candidates

    def _context_aware_filtering(self,
                               candidates: List[RescueCandidate],
                               context: Optional[Dict]) -> List[RescueCandidate]:
        """
        Контекстно-зависимая фильтрация
        """
        if not context:
            return candidates

        filtered = []

        for candidate in candidates:
            # Анализ контекста для принятия решения
            should_rescue = self._evaluate_context_rescue_decision(candidate, context)

            if should_rescue:
                filtered.append(candidate)

        return filtered

    def _evaluate_context_rescue_decision(self,
                                        candidate: RescueCandidate,
                                        context: Dict) -> bool:
        """
        Оценка решения о rescue на основе контекста
        """
        # Факторы, повышающие вероятность rescue
        positive_factors = 0

        # Если кластер маленький и лицо близко к порогу
        if (context.get('cluster_size', 1) <= 2 and
            candidate.original_score >= settings.quality_threshold * 0.8):
            positive_factors += 2

        # Если лицо единственное на изображении
        if context.get('image_faces_count', 1) == 1:
            positive_factors += 1

        # Если кластер имеет низкое качество
        if context.get('cluster_quality', 0.5) < 0.7:
            positive_factors += 1

        # Если есть высокая согласованность методов
        if candidate.validation_details.get('cross_validation_score', 0) > 0.8:
            positive_factors += 1

        # Факторы, понижающие вероятность rescue
        negative_factors = 0

        # Высокий риск
        if candidate.risk_level == 'high':
            negative_factors += 2

        # Очень низкое качество
        if candidate.original_score < settings.quality_threshold * 0.5:
            negative_factors += 1

        # Решение: положительные факторы должны перевешивать отрицательные
        return positive_factors > negative_factors

    def _attempt_rescue(self,
                       candidate: RescueCandidate,
                       img: np.ndarray,
                       context: Optional[Dict]) -> bool:
        """
        Попытка rescue кандидата
        """
        # Для простоты - используем потенциал rescue как вероятность успеха
        # В реальности здесь можно использовать более сложную логику
        success_probability = candidate.rescue_potential

        # Корректировка на основе стратегии
        if self.strategy == RescueStrategy.CONSERVATIVE:
            success_probability *= 0.8  # Более строгие требования
        elif self.strategy == RescueStrategy.AGGRESSIVE:
            success_probability *= 1.2  # Более лояльные требования

        # Корректировка на основе контекста
        if context and context.get('cluster_quality', 0.5) < 0.6:
            success_probability *= 1.1  # Плохие кластеры получают бонус

        # Случайное решение на основе вероятности
        # В продакшене можно использовать более детерминированную логику
        import random
        success = random.random() < success_probability

        if success:
            self.rescue_stats['successful_rescues'] += 1
        else:
            self.rescue_stats['failed_rescues'] += 1

        return success

    def _generate_rescue_recommendations(self,
                                       rescued_faces: List[Dict],
                                       rejected_faces: List[Dict],
                                       context: Optional[Dict]) -> List[str]:
        """
        Генерация рекомендаций на основе результатов rescue
        """
        recommendations = []

        rescue_rate = len(rescued_faces) / max(1, len(rescued_faces) + len(rejected_faces))

        if rescue_rate > 0.5:
            recommendations.append("Высокий уровень rescue - рассмотрите повышение quality_threshold")

        if rescue_rate < 0.1:
            recommendations.append("Низкий уровень rescue - возможно стоит смягчить критерии")

        if context and context.get('cluster_quality', 0.5) < 0.7:
            recommendations.append("Низкое качество кластеров - rescue помогает стабилизировать")

        if len(rescued_faces) > len(rejected_faces):
            recommendations.append("Большинство кандидатов rescued - проверьте параметры валидации")

        return recommendations

    def _update_learning_model(self,
                             rescued_faces: List[Dict],
                             rejected_faces: List[Dict]) -> None:
        """
        Обновление модели обучения на основе результатов
        """
        if not self.adaptive_learning:
            return

        # Сохраняем паттерны успешных rescue
        for face in rescued_faces:
            pattern = self._extract_success_pattern(face)
            self.success_patterns.append(pattern)

        # Сохраняем паттерны неудачных rescue
        for face in rejected_faces:
            if face.get('rescued_attempted', False):
                pattern = self._extract_failure_pattern(face)
                self.failure_patterns.append(pattern)

        # Ограничиваем размер истории
        max_history = 1000
        self.success_patterns = self.success_patterns[-max_history:]
        self.failure_patterns = self.failure_patterns[-max_history:]

    def _extract_success_pattern(self, face: Dict) -> Dict[str, Any]:
        """Извлечение паттерна успешного rescue"""
        return {
            'original_score': face.get('quality', 0.0),
            'rescue_info': face.get('rescue_info', {}),
            'validation_details': face.get('validation_details', {}),
            'outcome': 'success'
        }

    def _extract_failure_pattern(self, face: Dict) -> Dict[str, Any]:
        """Извлечение паттерна неудачного rescue"""
        return {
            'original_score': face.get('quality', 0.0),
            'validation_details': face.get('validation_details', {}),
            'outcome': 'failure'
        }

    def _classify_face_position(self, bbox: List[int], context: Dict) -> str:
        """Классификация позиции лица на изображении"""
        if not bbox or len(bbox) != 4:
            return 'unknown'

        img_height = context.get('image_height', 1000)
        img_width = context.get('image_width', 1000)

        x1, y1, x2, y2 = bbox
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2

        # Определение позиции
        if face_center_x < img_width * 0.3:
            x_pos = 'left'
        elif face_center_x > img_width * 0.7:
            x_pos = 'right'
        else:
            x_pos = 'center'

        if face_center_y < img_height * 0.3:
            y_pos = 'top'
        elif face_center_y > img_height * 0.7:
            y_pos = 'bottom'
        else:
            y_pos = 'middle'

        return f"{y_pos}_{x_pos}"

    def _compile_statistics(self, processing_time: float) -> Dict[str, Any]:
        """Компиляция статистики rescue операций"""
        stats = self.rescue_stats.copy()
        stats.update({
            'processing_time': processing_time,
            'success_rate': (stats['successful_rescues'] /
                           max(1, stats['successful_rescues'] + stats['failed_rescues'])),
            'rescue_efficiency': (stats['rescued_faces'] /
                                max(1, stats['total_candidates'])),
            'strategy': self.strategy.value,
            'learning_enabled': self.adaptive_learning
        })

        return stats

    def get_rescue_statistics(self) -> Dict[str, Any]:
        """Получение полной статистики rescue операций"""
        return {
            'current_stats': self._compile_statistics(0.0),
            'learning_patterns': {
                'success_patterns_count': len(self.success_patterns),
                'failure_patterns_count': len(self.failure_patterns),
                'recent_success_rate': self._calculate_recent_success_rate()
            },
            'strategy_performance': self.rescue_stats['by_strategy']
        }

    def _calculate_recent_success_rate(self) -> float:
        """Расчет recent success rate из паттернов обучения"""
        if not self.success_patterns:
            return 0.0

        recent_patterns = self.success_patterns[-100:]  # Последние 100
        success_count = sum(1 for p in recent_patterns if p['outcome'] == 'success')

        return success_count / len(recent_patterns)
