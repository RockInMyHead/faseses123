"""
Production-вариант кластеризации лиц на базе ArcFace + HDBSCAN.
- Детекция и эмбеддинги: InsightFace (ArcFace), app.FaceAnalysis
- Кластеризация: адаптивная плотностная HDBSCAN поверх L2-нормированных эмбеддингов
- Совместим по интерфейсу с упрощённой версией: build_plan_pro, distribute_to_folders, process_group_folder
- Устойчив к Unicode-путям, много-лицам на фото, копированию для мультикластерных кадров

Зависимости:
    pip install insightface onnxruntime opencv-python pillow scikit-learn numpy hdbscan

Автор: prod-ready скелет. Подключайте в своё приложение напрямую.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from collections import defaultdict

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import hdbscan  # type: ignore
except Exception as e:  # pragma: no cover
    hdbscan = None

try:
    from insightface.app import FaceAnalysis
except Exception as e:  # pragma: no cover
    FaceAnalysis = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
ProgressCB = Optional[Callable[[str, int], None]]

# Параметры качества распознавания (глобальные)
QUALITY_THRESHOLD = 0.75  # Базовый порог качества
MIN_FACE_SIZE = 64        # Минимальный размер лица в пикселях
MAX_FACE_SIZE = 512       # Максимальный размер лица в пикселях
MAX_FACE_ANGLE = 30       # Максимальный угол поворота лица (градусы)

# ------------------------
# Утилиты ввода/вывода
# ------------------------

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def validate_face_quality_dual(face: dict, img: np.ndarray) -> tuple[float, dict]:
    """
    Двойная валидация качества лица с детальным анализом.

    Returns:
        tuple: (final_score, validation_details)
            - final_score: float от 0.0 до 1.0
            - validation_details: dict с подробностями валидации
    """
    details = {
        'primary_score': 0.0,
        'secondary_score': 0.0,
        'cross_validation_score': 0.0,
        'final_score': 0.0,
        'method': 'dual_validation',
        'quality_metrics': {}
    }

    # Первичная валидация (основная функция)
    primary_score = validate_face_quality(face, img)
    details['primary_score'] = primary_score

    # Вторичная валидация (альтернативные метрики)
    secondary_score = validate_face_quality_alternative(face, img)
    details['secondary_score'] = secondary_score

    # Кросс-валидационная оценка (согласованность результатов)
    cross_validation_score = 1.0 - min(abs(primary_score - secondary_score), 0.5)
    details['cross_validation_score'] = cross_validation_score

    # Финальный скор с учетом кросс-валидации
    # Если модели согласны - повышаем уверенность
    # Если не согласны - понижаем уверенность
    agreement_factor = cross_validation_score  # 0.5-1.0
    average_score = (primary_score + secondary_score) / 2.0

    final_score = average_score * agreement_factor
    details['final_score'] = final_score

    # Детальные метрики качества
    details['quality_metrics'] = _extract_quality_metrics(face, img)

    return final_score, details


def validate_face_quality_alternative(face: dict, img: np.ndarray) -> float:
    """
    Альтернативная валидация качества лица с другими весами и метриками.
    """
    scores = []
    weights = []

    # 1. Встроенный score модели (больший вес)
    if 'det_score' in face:
        det_score = float(face['det_score'])
        scores.append(min(det_score * 2.5, 1.0))  # Более строгая нормализация
        weights.append(0.5)  # Большой вес

    # 2. Размер лица (меньший вес)
    bbox = face.get('bbox', [])
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        face_width = abs(x2 - x1)
        face_height = abs(y2 - y1)
        min_size = min(face_width, face_height)

        if min_size < MIN_FACE_SIZE:
            return 0.0

        # Более строгие требования к размеру
        if min_size < 100:
            size_score = min_size / 100.0
        elif min_size > 300:
            size_score = max(0.3, 1.0 - (min_size - 300) / 200.0)
        else:
            size_score = 1.0

        scores.append(size_score)
        weights.append(0.2)

    # 3. Качество изображения (больший вес для размытости)
    if len(bbox) == 4:
        blur_score = calculate_blur_score(img, bbox)
        scores.append(blur_score)
        weights.append(0.4)  # Большой вес для размытости

    # 4. Освещенность (стандартный вес)
    if len(bbox) == 4:
        brightness_score = calculate_brightness_score(img, bbox)
        scores.append(brightness_score)
        weights.append(0.3)

    # Взвешенное среднее
    if not scores:
        return 0.5

    if weights and len(weights) == len(scores):
        final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    else:
        final_score = sum(scores) / len(scores)

    return min(final_score, 1.0)


def _rescue_low_quality_faces(rejected_faces: List[Dict], img: np.ndarray, img_path: Path) -> List[Dict]:
    """
    Устаревшая функция rescue - оставлена для совместимости.
    Используйте _advanced_rescue_low_quality_faces для новых реализаций.
    """
    return _advanced_rescue_low_quality_faces(rejected_faces, img, img_path, None)


def _advanced_rescue_low_quality_faces(rejected_faces: List[Dict], img: np.ndarray, img_path: Path, embedder) -> List[Dict]:
    """
    Продвинутая попытка спасти низкокачественные лица для кластеризации.

    Использует AdvancedFaceRescue для интеллектуального анализа и rescue.
    """
    try:
        from app.services.face_detection.advanced_rescue import AdvancedFaceRescue, RescueStrategy

        # Инициализация rescue системы
        rescue_system = AdvancedFaceRescue(
            strategy=RescueStrategy.BALANCED,
            adaptive_learning=True
        )

        # Подготовка контекста
        context = _prepare_rescue_context(img, img_path, embedder)

        # Выполнение rescue
        rescue_result = rescue_system.rescue_faces(rejected_faces, img, str(img_path), context)

        # Логирование результатов
        if rescue_result.rescued_faces:
            print(f"🔄 Advanced rescue: {len(rescue_result.rescued_faces)} faces rescued from {img_path}")
            for rec in rescue_result.recommendations[:2]:  # Показываем первые 2 рекомендации
                print(f"   💡 {rec}")

        # Возвращаем rescued лица
        return rescue_result.rescued_faces

    except ImportError:
        # Fallback к старой системе если AdvancedFaceRescue недоступен
        print("⚠️ AdvancedFaceRescue not available, using legacy rescue")
        return _legacy_rescue_low_quality_faces(rejected_faces, img, img_path)


def _legacy_rescue_low_quality_faces(rejected_faces: List[Dict], img: np.ndarray, img_path: Path) -> List[Dict]:
    """
    Legacy версия rescue для обратной совместимости
    """
    rescued_faces = []
    rescue_threshold = QUALITY_THRESHOLD * 0.6  # 60% от основного порога

    for face in rejected_faces:
        # Проверяем, есть ли детальная информация о валидации
        if 'validation_details' in face:
            details = face['validation_details']

            # Rescue критерии:
            # 1. Хотя бы один скор выше rescue_threshold
            # 2. Согласованность между моделями > 0.7
            primary_ok = details['primary_score'] >= rescue_threshold
            secondary_ok = details['secondary_score'] >= rescue_threshold
            agreement_ok = details['cross_validation_score'] > 0.7

            if (primary_ok or secondary_ok) and agreement_ok:
                # Помечаем как rescued
                face['rescued'] = True
                face['rescue_reason'] = f"primary:{details['primary_score']:.2f}, secondary:{details['secondary_score']:.2f}"
                rescued_faces.append(face)
                print(f"🔄 Rescued лицо на {img_path} (score: {details['final_score']:.2f})")

        else:
            # Для обычных лиц - просто проверяем основной скор
            quality_score = validate_face_quality(face, img)
            if quality_score >= rescue_threshold:
                face['rescued'] = True
                face['rescue_reason'] = f"basic_rescue:{quality_score:.2f}"
                rescued_faces.append(face)
                print(f"🔄 Rescued лицо на {img_path} (basic score: {quality_score:.2f})")

    return rescued_faces


def _prepare_rescue_context(img: np.ndarray, img_path: Path, embedder) -> Dict[str, Any]:
    """
    Подготовка контекста для rescue операций
    """
    context = {
        'image_height': img.shape[0] if len(img.shape) >= 2 else 0,
        'image_width': img.shape[1] if len(img.shape) >= 2 else 0,
        'image_faces_count': 0,  # Будет заполнено позже
        'cluster_size': 1,  # По умолчанию
        'cluster_quality': 0.5,  # По умолчанию
        'temporal_context': 'single_image'
    }

    # Если есть embedder, получаем дополнительную информацию
    if embedder and hasattr(embedder, 'extract'):
        try:
            # Быстрая оценка количества лиц
            test_faces = embedder.extract(img)
            context['image_faces_count'] = len(test_faces)

            # Оценка качества изображения
            if test_faces:
                avg_quality = sum(f.get('quality', 0.5) for f in test_faces) / len(test_faces)
                context['image_quality'] = avg_quality
        except Exception:
            pass

    return context


def _extract_quality_metrics(face: dict, img: np.ndarray) -> dict:
    """Извлечение детальных метрик качества"""
    metrics = {}

    # Метрики из DualFaceEmbedder
    if 'quality_details' in face:
        quality_details = face['quality_details']
        metrics.update({
            'detection_score': quality_details.get('detection_score', 0.0),
            'face_size_score': quality_details.get('face_size', 0.0),
            'blur_score': quality_details.get('blur_score', 0.0),
            'brightness_score': quality_details.get('brightness', 0.0),
            'overall_quality': quality_details.get('overall', 0.0)
        })

    # Источник детекции
    metrics['source'] = face.get('source', 'unknown')
    metrics['model'] = face.get('model', 'unknown')
    metrics['cross_validated'] = face.get('cross_validated', False)

    return metrics


def validate_face_quality(face: dict, img: np.ndarray) -> float:
    """
    Комплексная оценка качества лица.

    Returns:
        float: Оценка качества от 0.0 до 1.0
    """
    scores = []

    # 1. Встроенный score модели (если доступен)
    if 'det_score' in face:
        det_score = float(face['det_score'])
        scores.append(min(det_score * 2.0, 1.0))  # Нормализация к 0-1

    # 2. Размер лица
    bbox = face.get('bbox', [])
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        face_width = abs(x2 - x1)
        face_height = abs(y2 - y1)

        # Проверка размера
        min_size = min(face_width, face_height)
        if min_size < MIN_FACE_SIZE:
            return 0.0  # Слишком маленькое лицо
        if min_size > MAX_FACE_SIZE:
            return 0.0  # Слишком большое лицо

        # Оценка размера (оптимально 128-256 пикселей)
        size_score = 1.0
        if min_size < 128:
            size_score = min_size / 128.0
        elif min_size > 256:
            size_score = max(0.5, 1.0 - (min_size - 256) / 256.0)
        scores.append(size_score)

    # 3. Угол поворота (если доступен)
    if 'pose' in face:
        pose = face['pose']
        if len(pose) >= 3:
            yaw, pitch, roll = pose[:3]
            max_angle = max(abs(yaw), abs(pitch), abs(roll))
            if max_angle > MAX_FACE_ANGLE:
                return 0.0  # Лицо повернуто слишком сильно
            angle_score = 1.0 - (max_angle / 90.0)  # Чем меньше угол, тем лучше
            scores.append(angle_score)

    # 4. Качество изображения (размытость)
    if len(bbox) == 4:
        blur_score = calculate_blur_score(img, bbox)
        scores.append(blur_score)

    # 5. Освещенность
    if len(bbox) == 4:
        brightness_score = calculate_brightness_score(img, bbox)
        scores.append(brightness_score)

    # Итоговый score - среднее всех оценок
    if not scores:
        return 0.5  # Средний score, если нет оценок

    final_score = sum(scores) / len(scores)
    return min(final_score, 1.0)

def calculate_blur_score(img: np.ndarray, bbox: list) -> float:
    """
    Оценка размытости лица на изображении.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        face_region = img[y1:y2, x1:x2]

        if face_region.size == 0:
            return 0.0

        # Преобразование в grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region

        # Вычисление Laplacian variance (мера резкости)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Нормализация к 0-1 (хорошая резкость > 100)
        blur_score = min(laplacian_var / 100.0, 1.0)
        return blur_score

    except Exception:
        return 0.5  # Средний score при ошибке

def calculate_brightness_score(img: np.ndarray, bbox: list) -> float:
    """
    Оценка освещенности лица.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        face_region = img[y1:y2, x1:x2]

        if face_region.size == 0:
            return 0.5

        # Вычисление средней яркости
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_region

        brightness = np.mean(gray) / 255.0

        # Оптимальная яркость 0.3-0.7
        if 0.3 <= brightness <= 0.7:
            return 1.0
        elif brightness < 0.3:
            return brightness / 0.3  # Темно
        else:
            return (1.0 - brightness) / 0.3  # Светло

    except Exception:
        return 0.5  # Средний score при ошибке


def refine_clusters_two_stage(X: np.ndarray, owners: List[Path], labels: np.ndarray,
                             progress_callback: ProgressCB = None) -> Tuple[np.ndarray, List[Path], np.ndarray]:
    """
    Двухэтапная кластеризация для уточнения результатов.

    Args:
        X: Матрица эмбеддингов
        owners: Список путей к изображениям
        labels: Метки кластеров после первичной кластеризации

    Returns:
        Кортеж (X, owners, labels) с уточненными результатами
    """
    if len(set(labels.tolist()) - {-1}) < 2:
        return X, owners, labels  # Недостаточно кластеров для уточнения

    # Этап 1: Найти сомнительные кластеры
    suspicious_cluster_ids = identify_suspicious_clusters_simple(X, labels)

    if not suspicious_cluster_ids:
        return X, owners, labels

    if progress_callback:
        progress_callback(f"🔄 Перекластеризация {len(suspicious_cluster_ids)} сомнительных кластеров", 87)

    # Этап 2: Перекластеризовать сомнительные кластеры
    refined_X = []
    refined_owners = []
    refined_labels = []

    # Группировка по кластерам
    cluster_indices = {}
    for i, label in enumerate(labels):
        if label not in cluster_indices:
            cluster_indices[label] = []
        cluster_indices[label].append(i)

    # Обработать каждый кластер
    new_cluster_id = max(labels) + 1  # Начать с нового ID

    for cluster_id in sorted(set(labels.tolist()) - {-1}):
        indices = cluster_indices[cluster_id]

        if cluster_id in suspicious_cluster_ids:
            # Перекластеризовать сомнительный кластер
            sub_X = X[indices]
            sub_labels = recluster_suspicious_cluster_simple(sub_X)

            # Добавить субкластеры с новыми ID
            for sub_label in set(sub_labels.tolist()) - {-1}:
                sub_indices = [indices[i] for i, lbl in enumerate(sub_labels) if lbl == sub_label]
                for idx in sub_indices:
                    refined_X.append(X[idx])
                    refined_owners.append(owners[idx])
                    refined_labels.append(new_cluster_id)
                new_cluster_id += 1
        else:
            # Оставить надежный кластер как есть
            for idx in indices:
                refined_X.append(X[idx])
                refined_owners.append(owners[idx])
                refined_labels.append(cluster_id)

    # Преобразовать обратно в numpy массивы
    if refined_X:
        refined_X = np.vstack(refined_X)
        refined_labels = np.array(refined_labels)

        if progress_callback:
            progress_callback(f"✅ После уточнения: {len(set(refined_labels.tolist()) - {-1})} кластеров", 90)

        return refined_X, refined_owners, refined_labels
    else:
        return X, owners, labels


def identify_suspicious_clusters_simple(X: np.ndarray, labels: np.ndarray) -> Set[int]:
    """
    Найти сомнительные кластеры в простой кластеризации.
    """
    suspicious = set()
    unique_labels = set(labels.tolist()) - {-1}

    for cluster_id in unique_labels:
        # Найти индексы лиц в этом кластере
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]

        if len(cluster_indices) < 3:
            suspicious.add(cluster_id)
            continue

        # Проверить плотность кластера
        cluster_embeddings = X[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Критерии сомнительности
        if avg_distance > 0.4 or std_distance > 0.6:
            suspicious.add(cluster_id)

    return suspicious


def recluster_suspicious_cluster_simple(X: np.ndarray) -> np.ndarray:
    """
    Перекластеризовать сомнительный кластер с более строгими параметрами.
    """
    if X.shape[0] < 3:
        # Для маленьких кластеров вернуть все в один кластер
        return np.zeros(X.shape[0], dtype=np.int32)

    # Более строгие параметры
    strict_min_cluster_size = max(2, X.shape[0] // 4)  # Минимум 2 или 1/4 от размера

    labels = cluster_embeddings_hdbscan(
        X,
        min_cluster_size=strict_min_cluster_size,
        min_samples=strict_min_cluster_size,
    )

    return labels


def post_process_clusters_simple(X: np.ndarray, owners: List[Path], labels: np.ndarray,
                                progress_callback: ProgressCB = None) -> Tuple[np.ndarray, List[Path], np.ndarray]:
    """
    Расширенная пост-обработка кластеров для локальной кластеризации.

    Аналог глобальной пост-обработки, но для простых массивов.
    """
    if len(set(labels.tolist()) - {-1}) < 2:
        return X, owners, labels

    if progress_callback:
        progress_callback("🔧 Пост-обработка кластеров", 88)

    # Группировка по кластерам
    cluster_indices = {}
    for i, label in enumerate(labels):
        if label not in cluster_indices:
            cluster_indices[label] = []
        cluster_indices[label].append(i)

    # Анализ качества кластеров
    cluster_stats = analyze_clusters_quality_simple(X, labels, cluster_indices)

    # Удаление шумовых кластеров
    X, owners, labels = remove_noise_clusters_simple(X, owners, labels, cluster_indices)

    # Объединение близких кластеров
    X, owners, labels = merge_similar_clusters_simple(X, owners, labels, cluster_indices)

    # Финальная проверка
    final_check = len(set(labels.tolist()) - {-1})
    if progress_callback:
        progress_callback(f"✅ Пост-обработка завершена: {final_check} кластеров", 95)

    return X, owners, labels


def analyze_clusters_quality_simple(X: np.ndarray, labels: np.ndarray,
                                   cluster_indices: Dict[int, List[int]]) -> Dict:
    """
    Анализ качества кластеров в простой кластеризации.
    """
    stats = {}
    unique_labels = set(labels.tolist()) - {-1}

    for cluster_id in unique_labels:
        indices = cluster_indices[cluster_id]
        if len(indices) < 2:
            continue

        embeddings = X[indices]
        centroid = np.mean(embeddings, axis=0)
        distances = [np.linalg.norm(emb - centroid) for emb in embeddings]

        stats[cluster_id] = {
            'size': len(indices),
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances)
        }

    return stats


def remove_noise_clusters_simple(X: np.ndarray, owners: List[Path], labels: np.ndarray,
                                cluster_indices: Dict[int, List[int]]) -> Tuple[np.ndarray, List[Path], np.ndarray]:
    """
    Удалить шумовые кластеры из простой кластеризации.
    """
    min_cluster_size = 2
    valid_indices = []

    for i, label in enumerate(labels):
        if label == -1:
            continue  # Шум всегда удаляем

        cluster_size = len(cluster_indices.get(label, []))
        if cluster_size >= min_cluster_size:
            valid_indices.append(i)

    if not valid_indices:
        return X, owners, labels

    # Фильтровать массивы
    X_filtered = X[valid_indices]
    owners_filtered = [owners[i] for i in valid_indices]
    labels_filtered = labels[valid_indices]

    return X_filtered, owners_filtered, labels_filtered


def merge_similar_clusters_simple(X: np.ndarray, owners: List[Path], labels: np.ndarray,
                                 cluster_indices: Dict[int, List[int]]) -> Tuple[np.ndarray, List[Path], np.ndarray]:
    """
    Объединить похожие кластеры в простой кластеризации.
    """
    unique_labels = sorted(set(labels.tolist()) - {-1})
    if len(unique_labels) < 2:
        return X, owners, labels

    # Вычислить центроиды кластеров
    centroids = {}
    for cluster_id in unique_labels:
        indices = cluster_indices[cluster_id]
        centroids[cluster_id] = np.mean(X[indices], axis=0)

    # Найти пары для объединения
    merge_threshold = 0.75  # Более строгий порог для локальной кластеризации
    to_merge = []

    for i, cid1 in enumerate(unique_labels):
        for cid2 in unique_labels[i+1:]:
            c1 = centroids[cid1]
            c2 = centroids[cid2]

            similarity = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
            if similarity > merge_threshold:
                to_merge.append((cid1, cid2))

    # Объединить найденные кластеры
    for cid1, cid2 in reversed(to_merge):
        # Заменить все метки cid2 на cid1
        labels = np.where(labels == cid2, cid1, labels)

    return X, owners, labels


def imread_safe(path: Path) -> Optional[np.ndarray]:
    """Аккуратное чтение изображений (BGR->RGB). Возвращает None при ошибке.
    Используем cv2.imdecode для лучшей поддержки Unicode путей.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception:
        return None


# ------------------------
# Инициализация модели InsightFace
# ------------------------
@dataclass
class ArcFaceConfig:
    det_size: Tuple[int, int] = (640, 640)
    ctx_id: int = 0                   # GPU: индекс, CPU: -1
    allowed_blur: float = 0.8         # порог качества (примерный, отфильтруем явный мусор)


class ArcFaceEmbedder:
    def __init__(self, config: ArcFaceConfig = ArcFaceConfig(), model_name: str = "buffalo_l"):
        if FaceAnalysis is None:
            raise ImportError("insightface не установлен. Установите пакет insightface.")
        self.app = FaceAnalysis(name=model_name)
        # ctx_id=-1 → CPU, иначе GPU. det_size влияет на recall/скорость детектора
        self.app.prepare(ctx_id=config.ctx_id, det_size=config.det_size)
        self.allowed_blur = config.allowed_blur

    def extract(self, img_rgb: np.ndarray) -> List[Dict]:
        """Возвращает список лиц: [{embedding, quality, bbox}]. embedding уже L2-нормирован InsightFace."""
        faces = self.app.get(img_rgb)
        results: List[Dict] = []
        for f in faces:
            # f.normed_embedding — L2-нормированный эмбеддинг (512,)
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                # запасной путь: normalise raw embedding
                raw = getattr(f, "embedding", None)
                if raw is None:
                    continue
                v = np.asarray(raw, dtype=np.float32)
                n = np.linalg.norm(v) + 1e-12
                emb = (v / n).astype(np.float32)
            else:
                emb = np.asarray(emb, dtype=np.float32)

            # эвристика качества: используем blur/pose/детскую confidence если есть
            quality = float(getattr(f, "det_score", 0.99))
            if quality <= 0:  # страховка
                quality = 0.99

            bbox = tuple(int(x) for x in f.bbox.astype(int).tolist())
            results.append({
                "embedding": emb,
                "quality": quality,
                "bbox": bbox,
            })
        return results


def cluster_embeddings_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    """Кластеризация эмбеддингов с адаптивным HDBSCAN."""
    if embeddings.size == 0:
        return np.array([], dtype=np.int32)
    if hdbscan is None:
        raise ImportError("hdbscan не установлен. Установите пакет hdbscan.")

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    X = embeddings / norms

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    uniq = sorted(x for x in set(labels.tolist()) if x != -1)
    remap = {old: i for i, old in enumerate(uniq)}
    out = labels.copy()
    for i, lb in enumerate(labels):
        out[i] = remap.get(int(lb), -1)
    return out


# ------------------------
# Основной пайплайн
# ------------------------

def build_plan_pro(
    input_dir: Path,
    progress_callback: ProgressCB = None,
    sim_threshold: float = 0.60,
    min_cluster_size: int = 2,
    ctx_id: int = 0,
    det_size: Tuple[int, int] = (640, 640),
    model_name: str = "buffalo_l",
    min_samples: Optional[int] = None,
    joint_mode: str = "copy",
) -> Dict:
    # sim_threshold сохраняем для обратной совместимости — HDBSCAN его не использует.
    """Production-кластеризация лиц с ArcFace + Faiss.

    Возвращает dict:
      {
        "clusters": {"0": ["/abs/path/img1.jpg", ...], ...},
        "plan": [ {"path": str, "cluster": [int, ...], "faces": int}, ...],
        "unreadable": [str, ...],
        "no_faces": [str, ...]
      }
    """
    t0 = time.time()
    input_dir = Path(input_dir)
    if progress_callback:
        progress_callback(f"🚀 Кластеризация: {input_dir}", 2)

    # Инициализация эмбеддера
    # Для buffalo_l используем меньший det_size если память ограничена
    if model_name == "buffalo_l":
        # Пробуем оптимизировать для buffalo_l
        try:
            emb = ArcFaceEmbedder(ArcFaceConfig(det_size=det_size, ctx_id=ctx_id), model_name=model_name)
        except Exception as e:
            print(f"Warning: buffalo_l failed with det_size {det_size}, trying smaller...")
            # Если не получается, пробуем с меньшим размером
            smaller_det_size = (max(320, det_size[0] // 2), max(320, det_size[1] // 2))
            emb = ArcFaceEmbedder(ArcFaceConfig(det_size=smaller_det_size, ctx_id=ctx_id), model_name=model_name)
            print(f"Using buffalo_l with reduced det_size: {smaller_det_size}")
    else:
        emb = ArcFaceEmbedder(ArcFaceConfig(det_size=det_size, ctx_id=ctx_id), model_name=model_name)

    # Сбор изображений
    all_images = [p for p in input_dir.rglob("*") if p.is_file() and is_image(p)]
    if progress_callback:
        progress_callback(f"📂 Найдено изображений: {len(all_images)}", 5)

    owners: List[Path] = []
    all_embeddings: List[np.ndarray] = []
    img_face_count: Dict[Path, int] = {}
    unreadable: List[Path] = []
    no_faces: List[Path] = []

    total = len(all_images)
    for i, img_path in enumerate(all_images):
        if progress_callback and (i % 10 == 0):
            percent = 5 + int((i + 1) / max(1, total) * 60)
            progress_callback(f"📷 Анализ {i+1}/{total}", percent)

        img = imread_safe(img_path)
        if img is None:
            unreadable.append(img_path)
            continue

        faces = emb.extract(img)
        if not faces:
            no_faces.append(img_path)
            continue

        # Валидация качества лиц
        validated_faces = []
        rejected_faces = []

        for face in faces:
            # Определяем тип данных и выбираем метод валидации
            if 'quality_details' in face and 'cross_validated' in face:
                # Данные от DualFaceEmbedder - используем двойную валидацию
                quality_score, validation_details = validate_face_quality_dual(face, img)
                face['validation_details'] = validation_details  # Сохраняем детали для анализа
            else:
                # Обычные данные - используем стандартную валидацию
                quality_score = validate_face_quality(face, img)

            if quality_score >= QUALITY_THRESHOLD:
                validated_faces.append(face)
            else:
                rejected_faces.append(face)
                print(f"⚠️ Пропущено лицо низкого качества (score={quality_score:.2f}) на {img_path}")
                if 'validation_details' in face:
                    details = face['validation_details']
                    print(f"   └─ Первичный: {details['primary_score']:.2f}, "
                          f"Вторичный: {details['secondary_score']:.2f}, "
                          f"Согласованность: {details['cross_validation_score']:.2f}")

        # Если нет валидных лиц, но есть отклоненные - попробуем rescue низкокачественные
        if not validated_faces and rejected_faces:
            rescued_faces = _advanced_rescue_low_quality_faces(rejected_faces, img, img_path, emb)
            validated_faces.extend(rescued_faces)

        if not validated_faces:
            no_faces.append(img_path)
            continue

        img_face_count[img_path] = len(validated_faces)
        for face in validated_faces:
            all_embeddings.append(face["embedding"])  # уже L2-норм
            owners.append(img_path)

    if not all_embeddings:
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
        }

    X = np.vstack(all_embeddings).astype(np.float32)

    # Кластеризация через HDBSCAN
    if progress_callback:
        progress_callback("🔗 Кластеризация HDBSCAN", 70)
    labels = cluster_embeddings_hdbscan(
        X,
        min_cluster_size=max(2, min_cluster_size),
        min_samples=min_samples,
    )

    if progress_callback:
        progress_callback(f"✅ Кластеров: {len(set(labels.tolist()) - {-1})}", 85)

    # Двухэтапная кластеризация для уточнения результатов
    X, owners, labels = refine_clusters_two_stage(X, owners, labels, progress_callback)

    # Расширенная пост-обработка кластеров
    X, owners, labels = post_process_clusters_simple(X, owners, labels, progress_callback)

    # Формирование мапов
    cluster_map: Dict[int, set[Path]] = defaultdict(set)
    cluster_by_img: Dict[Path, set[int]] = defaultdict(set)

    for lb, path in zip(labels, owners):
        if lb == -1:
            # одиночки: можно поместить в отдельную папку "-1" либо пропустить из плана
            continue
        cluster_map[int(lb)].add(path)
        cluster_by_img[path].add(int(lb))

    # План перемещений/копирования
    plan: List[Dict] = []
    for path in all_images:
        cl = cluster_by_img.get(path)
        if not cl:
            continue
        plan.append({
            "path": str(path),
            "cluster": sorted(list(cl)),
            "faces": img_face_count.get(path, 0),
        })

    if progress_callback:
        dt = time.time() - t0
        progress_callback(f"⏱️ Обработка завершена за {dt:.1f}с", 95)

    return {
        "clusters": {str(k): [str(p) for p in sorted(v)] for k, v in cluster_map.items()},
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
    }


# ------------------------
# Распределение по папкам (совместимо с упрощённой версией)
# ------------------------

def distribute_to_folders(plan: dict, base_dir: Path, cluster_start: int = 1, progress_callback: ProgressCB = None, common_mode: bool = False, joint_mode: str = "copy", post_validate: bool = False) -> Tuple[int, int, int]:
    import shutil

    moved, copied = 0, 0
    moved_paths = set()

    used_clusters = sorted({c for item in plan.get("plan", []) for c in item["cluster"]})
    # В режиме ОБЩАЯ получаем только кластеры людей с общих фотографий
    common_photo_clusters = set()
    if common_mode:
        # Находим кластеры людей, которые есть на общих фотографиях
        for item in plan.get("plan", []):
            src = Path(item["path"])
            is_common_photo = any(excluded_name in str(src.parent).lower() for excluded_name in EXCLUDED_COMMON_NAMES)
            if is_common_photo:
                common_photo_clusters.update(item["cluster"])

        # Объединяем с used_clusters только кластеры с общих фото
        used_clusters = sorted(set(used_clusters) | common_photo_clusters)

    # В режиме common_mode сохраняем реальные номера кластеров, иначе перенумеровываем
    if common_mode:
        cluster_id_map = {old: old for old in used_clusters}  # Сохраняем реальные номера
    else:
        cluster_id_map = {old: cluster_start + idx for idx, old in enumerate(used_clusters)}

    plan_items = plan.get("plan", [])
    total_items = len(plan_items)
    if progress_callback:
        progress_callback(f"🔄 Распределение {total_items} файлов по папкам...", 0)

    cluster_file_counts: Dict[int, int] = {}
    for item in plan_items:
        src = Path(item["path"])
        # Проверяем, является ли файл общим (находится в папке "общие")
        is_common_photo = any(excluded_name in str(src.parent).lower() for excluded_name in EXCLUDED_COMMON_NAMES)
        
        if not is_common_photo:  # Считаем только НЕ общие фотографии
            clusters = [cluster_id_map[c] for c in item["cluster"]]
            for cid in clusters:
                cluster_file_counts[cid] = cluster_file_counts.get(cid, 0) + 1

    for i, item in enumerate(plan_items):
        if progress_callback:
            percent = int((i + 1) / max(total_items, 1) * 100)
            progress_callback(f"📁 Распределение файлов: {percent}% ({i+1}/{total_items})", percent)

        src = Path(item["path"])  # исходный файл
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        if not src.exists():
            continue
            
        # Проверяем, является ли файл общим (находится в папке "общие")
        is_common_photo = any(excluded_name in str(src.parent).lower() for excluded_name in EXCLUDED_COMMON_NAMES)
        
        if is_common_photo:
            # Общие фотографии НЕ перемещаем - оставляем на месте
            print(f"📌 Общая фотография оставлена: {src.name}")
            continue

        if len(clusters) == 1:
            # Определяем папку назначения: берем родительскую папку файла
            parent_folder = src.parent
            dst = parent_folder / f"{clusters[0]}" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve():
                shutil.move(str(src), str(dst))
                moved += 1
                moved_paths.add(src.parent)
        else:
            # Обработка мульти-кластерных файлов в зависимости от режима
            parent_folder = src.parent
            if joint_mode == "combine":
                # Создаем комбинированную папку
                combo_name = "+".join(str(c) for c in sorted(clusters))
                dst = parent_folder / combo_name / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.resolve() != dst.resolve():
                    shutil.move(str(src), str(dst))
                    moved += 1
                    moved_paths.add(src.parent)
            else:  # joint_mode == "copy" (по умолчанию)
                # Копируем в каждую папку кластера
                for cid in clusters:
                    dst = parent_folder / f"{cid}" / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.resolve() != dst.resolve():
                        shutil.copy2(str(src), str(dst))
                        copied += 1
                try:
                    src.unlink()
                except Exception:
                    pass

    # Очистка пустых папок
    if progress_callback:
        progress_callback("🧹 Очистка пустых папок...", 95)

    # Собираем все уникальные родительские папки из перемещенных файлов
    parent_folders = set()
    for item in plan_items:
        src = Path(item["path"])
        if src.parent.exists():
            parent_folders.add(src.parent)

    # Удаляем пустые папки кластеров
    for parent_folder in parent_folders:
        for cid in cluster_file_counts.keys():
            folder_path = parent_folder / str(cid)
            if folder_path.exists():
                # Считаем реальное количество файлов в папке
                real_count = 0
                for file_path in folder_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        real_count += 1

                if real_count == 0:
                    # Удаляем пустые папки
                    try:
                        folder_path.rmdir()
                        print(f"🗑️ Удалена пустая папка: {folder_path}")
                    except Exception:
                        pass

    # Чистим пустые каталоги
    if progress_callback:
        progress_callback("🧹 Очистка пустых папок...", 100)
    for p in sorted(moved_paths, key=lambda x: len(str(x)), reverse=True):
        try:
            p.rmdir()
        except Exception:
            pass

    # В режиме ОБЩАЯ создаем пустые папки для всех найденных кластеров + 2 дополнительные
    if common_mode:
        # Создаем папки в родительской папке (не в папке "общие")
        parent_dir = base_dir.parent
        print(f"📁 Создаем папки в родительской директории: {parent_dir}")

        # Создаем папки для всех найденных кластеров (теперь с реальными номерами)
        for cluster_id in used_clusters:
            empty_folder = parent_dir / str(cluster_id)
            empty_folder.mkdir(parents=True, exist_ok=True)
            print(f"📁 Создана пустая папка для кластера: {cluster_id} в {parent_dir}")

        # Создаем 2 дополнительные пустые папки
        max_cluster_id = max(used_clusters) if used_clusters else 0
        for i in range(1, 3):  # Создаем 2 дополнительные папки
            extra_cluster_id = max_cluster_id + i
            extra_folder = parent_dir / str(extra_cluster_id)
            extra_folder.mkdir(parents=True, exist_ok=True)
            print(f"📁 Создана дополнительная пустая папка: {extra_cluster_id} в {parent_dir}")

    # Пост-валидация кластеров
    if post_validate:
        print(f"🔍 [POST-VALIDATE] Начинаем пост-валидацию, post_validate={post_validate}")
        print(f"🔍 [POST-VALIDATE] Количество кластеров для проверки: {len(cluster_file_counts)}")
        if progress_callback:
            progress_callback("🔍 Пост-валидация кластеров...", 95)
        try:
            false_positives_moved = post_validate_clusters(base_dir, cluster_file_counts.keys(), progress_callback)
            print(f"✅ [POST-VALIDATE] Пост-валидация завершена, возвращено {false_positives_moved} фото")
            if false_positives_moved > 0:
                print(f"⚠️ Возвращено {false_positives_moved} фото из false positive кластеров в родительскую папку")
        except Exception as e:
            print(f"❌ [POST-VALIDATE] Критическая ошибка в пост-валидации: {e}")
            import traceback
            traceback.print_exc()

    return moved, copied, cluster_start + len(used_clusters)


# ------------------------
# Пост-валидация кластеров
# ------------------------

def post_validate_clusters(base_dir: Path, cluster_ids: Iterable[int], progress_callback: ProgressCB = None) -> int:
    """Проверяет созданные кластеры на false positives.
    Если в папке кластера оказались лица разных людей, возвращает фото в родительскую папку.

    Возвращает количество перемещенных фото.
    """
    print(f"🔍 [POST-VALIDATE] Функция post_validate_clusters запущена")
    print(f"🔍 [POST-VALIDATE] base_dir: {base_dir}")
    print(f"🔍 [POST-VALIDATE] cluster_ids: {list(cluster_ids)}")

    total_moved = 0
    checked_clusters = 0

    for cluster_id in cluster_ids:
        print(f"🔍 [POST-VALIDATE] Проверяем кластер {cluster_id}")
        cluster_dir = base_dir / str(cluster_id)
        if not cluster_dir.exists():
            continue

        # Собираем все фото из кластерной папки
        cluster_images = []
        for file_path in cluster_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMG_EXTS:
                cluster_images.append(file_path)

        print(f"🔍 [POST-VALIDATE] В кластере {cluster_id} найдено {len(cluster_images)} фото: {[p.name for p in cluster_images[:3]]}{'...' if len(cluster_images) > 3 else ''}")

        if len(cluster_images) < 2:
            # Папка с 0-1 фото не проверяем
            print(f"🔍 [POST-VALIDATE] Кластер {cluster_id} пропущен (слишком мало фото)")
            continue

        checked_clusters += 1
        print(f"🔍 [POST-VALIDATE] Начинаем валидацию кластера {cluster_id} с {len(cluster_images)} фото")
        if progress_callback and checked_clusters % 5 == 0:
            progress_callback(f"🔍 Проверено {checked_clusters} кластеров...", 95)

        # Повторно кластеризуем лица из этой папки
        try:
            print(f"🔍 [POST-VALIDATE] Запускаем повторную кластеризацию для кластера {cluster_id}")
            # Создаем временный план только для этой папки
            temp_plan = build_plan_pro(
                cluster_dir,
                progress_callback=None,  # Тихая обработка
                sim_threshold=0.6,
                min_cluster_size=2,
                ctx_id=0,
                det_size=(640, 640),
                joint_mode="copy"
            )

            # Проверяем результат
            clusters_in_folder = temp_plan.get("clusters", {})
            print(f"🔍 [POST-VALIDATE] Результат повторной кластеризации кластера {cluster_id}: {len(clusters_in_folder)} кластеров")
            if len(clusters_in_folder) > 1:
                # False positive! В папке оказались разные люди
                print(f"⚠️ [POST-VALIDATE] False positive в кластере {cluster_id}: найдено {len(clusters_in_folder)} разных людей")
                print(f"⚠️ [POST-VALIDATE] Кластеры в папке: {list(clusters_in_folder.keys())}")

                # Возвращаем все фото обратно в родительскую папку (base_dir)
                for img_path in cluster_images:
                    dst = base_dir / img_path.name
                    counter = 1
                    while dst.exists():
                        stem = img_path.stem
                        suffix = img_path.suffix
                        dst = base_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                    shutil.move(str(img_path), str(dst))
                    total_moved += 1
                    print(f"↩️ [POST-VALIDATE] Фото {img_path.name} возвращено в {base_dir}")

                # Удаляем пустую папку кластера
                try:
                    cluster_dir.rmdir()
                    print(f"🗑️ [POST-VALIDATE] Удалена false positive папка: {cluster_dir}")
                except Exception as e:
                    print(f"⚠️ [POST-VALIDATE] Не удалось удалить папку {cluster_dir}: {e}")

        except Exception as e:
            print(f"⚠️ [POST-VALIDATE] Ошибка валидации кластера {cluster_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if progress_callback:
        progress_callback(f"✅ Пост-валидация завершена: проверено {checked_clusters} кластеров, возвращено {total_moved} фото", 100)

    return total_moved


# ------------------------
# Групповая обработка и «общие» папки
# ------------------------

EXCLUDED_COMMON_NAMES = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]


def find_common_folders_recursive(group_dir: Path) -> List[Path]:
    common: List[Path] = []
    print(f"🔍 Ищем общие папки в: {group_dir}")
    for subdir in group_dir.rglob("*"):
        if subdir.is_dir():
            print(f"🔍 Проверяем папку: {subdir.name}")
            if any(ex in subdir.name.lower() for ex in EXCLUDED_COMMON_NAMES):
                print(f"✅ Найдена общая папка: {subdir}")
                common.append(subdir)
    print(f"📁 Найдено общих папок: {len(common)}")
    return common


def process_common_folder_at_level(common_dir: Path, progress_callback: ProgressCB = None,
                                   sim_threshold: float = 0.60, min_cluster_size: int = 2,
                                   ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)) -> Tuple[int, int]:
    """Обработка «общих» папок: раскладываем лица по подпапкам внутри самой «общей».
    Например: common/ → common/1 (...), common/2 (...)
    Возвращает (moved, copied).
    """
    data = build_plan_pro(common_dir, progress_callback=progress_callback,
                          sim_threshold=sim_threshold, min_cluster_size=min_cluster_size,
                          ctx_id=ctx_id, det_size=det_size)
    moved, copied, _ = distribute_to_folders(data, common_dir, cluster_start=1, progress_callback=progress_callback, common_mode=True)
    return moved, copied


def process_group_folder(group_dir: Path, progress_callback: ProgressCB = None,
                         include_excluded: bool = False,
                         sim_threshold: float = 0.60, min_cluster_size: int = 2,
                         ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)) -> Tuple[int, int, int]:
    """Обрабатывает группу подпапок: кластеризует каждую подпапку отдельно.

    Если include_excluded=False — папки из EXCLUDED_COMMON_NAMES пропускаются.
    Возвращает (moved_total, copied_total, next_cluster_counter).
    """
    group_dir = Path(group_dir)

    if include_excluded:
        commons = find_common_folders_recursive(group_dir)
        for i, c in enumerate(commons):
            if progress_callback:
                progress_callback(f"📋 Общие: {c.name} ({i+1}/{len(commons)})", 5 + int(i / max(1, len(commons)) * 20))
            process_common_folder_at_level(c, progress_callback=progress_callback,
                                           sim_threshold=sim_threshold, min_cluster_size=min_cluster_size,
                                           ctx_id=ctx_id, det_size=det_size)

    subdirs = [d for d in sorted(group_dir.iterdir()) if d.is_dir()]
    if not include_excluded:
        subdirs = [d for d in subdirs if all(ex not in d.name.lower() for ex in EXCLUDED_COMMON_NAMES)]

    total = len(subdirs)
    moved_all, copied_all = 0, 0
    next_cluster_id = 1  # Глобальный счетчик кластеров для всей группы

    for i, sub in enumerate(subdirs):
        if progress_callback:
            progress_callback(f"🔍 {sub.name}: кластеризация ({i+1}/{total})", 25 + int(i / max(1, total) * 60))
        data = build_plan_pro(
            input_dir=sub,
            progress_callback=progress_callback,
            sim_threshold=sim_threshold,
            min_cluster_size=min_cluster_size,
            ctx_id=ctx_id,
            det_size=det_size,
        )
        m, c, next_cluster_id = distribute_to_folders(data, sub, cluster_start=next_cluster_id, progress_callback=progress_callback)
        moved_all += m
        copied_all += c

    return moved_all, copied_all, next_cluster_id


# ------------------------
# CLI-обвязка (опционально)
# ------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ArcFace+Faiss face clustering")
    parser.add_argument("input", type=str, help="Папка с изображениями или группа папок")
    parser.add_argument("--group", action="store_true", help="Обрабатывать как группу подпапок")
    parser.add_argument("--include-common", action="store_true", help="Обрабатывать папки 'общие' внутри группы")
    parser.add_argument("--sim", type=float, default=0.60, help="Порог косинусной близости [0..1]")
    parser.add_argument("--minsz", type=int, default=2, help="Мин. размер кластера")
    parser.add_argument("--cpu", action="store_true", help="Принудительно CPU (ctx_id=-1)")
    parser.add_argument("--det", type=int, nargs=2, default=[640, 640], help="Размер детектора WxH")

    args = parser.parse_args()

    def cb(msg: str, p: int):
        print(f"[{p:3d}%] {msg}")

    if args.group:
        moved, copied, _ = process_group_folder(
            Path(args.input), progress_callback=cb,
            include_excluded=args.include_common,
            sim_threshold=args.sim, min_cluster_size=args.minsz,
            ctx_id=(-1 if args.cpu else 0), det_size=tuple(args.det),
        )
        print(f"DONE: moved={moved}, copied={copied}")
    else:
        data = build_plan_pro(
            Path(args.input), progress_callback=cb,
            sim_threshold=args.sim, min_cluster_size=args.minsz,
            ctx_id=(-1 if args.cpu else 0), det_size=tuple(args.det),
        )
        m, c, _ = distribute_to_folders(data, Path(args.input), cluster_start=1, progress_callback=cb)
        print(f"DONE: moved={m}, copied={c}")
