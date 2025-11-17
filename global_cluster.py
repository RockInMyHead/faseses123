"""
Глобальная кластеризация лиц - решение проблем локальной кластеризации.

Вместо кластеризации каждой папки отдельно:
- Собирает все лица из всех папок
- Кластеризует глобально
- Распределяет фото правильно по всем папкам

Автор: Global clustering solution
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

try:
    import hdbscan  # type: ignore
except Exception as e:  # pragma: no cover
    hdbscan = None

try:
    from insightface.app import FaceAnalysis
except Exception as e:  # pragma: no cover
    FaceAnalysis = None

# Импорт из существующего модуля
from cluster_simple import (
    IMG_EXTS, is_image, imread_safe, ArcFaceEmbedder, ArcFaceConfig,
    cluster_embeddings_hdbscan, EXCLUDED_COMMON_NAMES
)

ProgressCB = Optional[Callable[[str, int], None]]

# ------------------------
# Глобальные структуры данных
# ------------------------

@dataclass
class GlobalFace:
    """Глобальное представление лица"""
    embedding: np.ndarray
    image_path: Path
    folder_name: str  # имя папки (Младшая, Средняя, etc.)
    face_idx: int     # индекс лица на фото (если несколько лиц)

@dataclass
class GlobalCluster:
    """Глобальный кластер - один человек во всем проекте"""
    cluster_id: int
    faces: List[GlobalFace]
    folders: Set[str]  # все папки, где есть этот человек

    @property
    def photo_paths(self) -> Set[Path]:
        """Все уникальные фото этого человека"""
        return {face.image_path for face in self.faces}

    @property
    def folder_count(self) -> int:
        """Количество папок, где есть этот человек"""
        return len(self.folders)

# ------------------------
# Основной класс глобальной кластеризации
# ------------------------

class GlobalFaceCluster:
    """
    Глобальная кластеризация лиц по всему проекту.

    Решает проблемы:
    - Один человек получает одинаковый номер во всех папках
    - Совместные фото правильно распределяются
    - Нет дублирования кластеров
    """

    def __init__(self, config: ArcFaceConfig = ArcFaceConfig()):
        if FaceAnalysis is None:
            raise ImportError("insightface не установлен")

        self.config = config
        self.embedder = ArcFaceEmbedder(config)
        self.all_faces: List[GlobalFace] = []
        self.global_clusters: List[GlobalCluster] = []
        self.folder_names: Set[str] = set()

        # Параметры качества распознавания
        self.quality_threshold = 0.75  # Базовый порог качества
        self.min_face_size = 64        # Минимальный размер лица в пикселях
        self.max_face_size = 512       # Максимальный размер лица в пикселях
        self.max_face_angle = 30       # Максимальный угол поворота лица (градусы)

    def _validate_face_quality(self, face: dict, img: np.ndarray) -> float:
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
            if min_size < self.min_face_size:
                return 0.0  # Слишком маленькое лицо
            if min_size > self.max_face_size:
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
                if max_angle > self.max_face_angle:
                    return 0.0  # Лицо повернуто слишком сильно
                angle_score = 1.0 - (max_angle / 90.0)  # Чем меньше угол, тем лучше
                scores.append(angle_score)

        # 4. Качество изображения (размытость)
        if len(bbox) == 4:
            blur_score = self._calculate_blur_score(img, bbox)
            scores.append(blur_score)

        # 5. Освещенность
        if len(bbox) == 4:
            brightness_score = self._calculate_brightness_score(img, bbox)
            scores.append(brightness_score)

        # Итоговый score - среднее всех оценок
        if not scores:
            return 0.5  # Средний score, если нет оценок

        final_score = sum(scores) / len(scores)
        return min(final_score, 1.0)

    def _calculate_blur_score(self, img: np.ndarray, bbox: list) -> float:
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

    def _calculate_brightness_score(self, img: np.ndarray, bbox: list) -> float:
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

    def collect_faces_from_folders(self, root_dir: Path, progress_callback: ProgressCB = None) -> None:
        """
        Собрать все лица из всех подпапок.

        Args:
            root_dir: Корневая папка с подпапками (Младшая, Средняя, etc.)
        """
        root_dir = Path(root_dir)

        # Найти все подпапки (исключая общие)
        subdirs = []
        for item in root_dir.iterdir():
            if item.is_dir() and all(ex not in item.name.lower() for ex in EXCLUDED_COMMON_NAMES):
                subdirs.append(item)

        if progress_callback:
            progress_callback(f"🔍 Найдено папок: {len(subdirs)}", 5)

        total_images = 0
        for subdir in subdirs:
            images = [p for p in subdir.rglob("*") if p.is_file() and is_image(p)]
            total_images += len(images)

        if progress_callback:
            progress_callback(f"📸 Всего изображений: {total_images}", 10)

        processed_images = 0
        for subdir in subdirs:
            folder_name = subdir.name
            self.folder_names.add(folder_name)

            images = [p for p in subdir.rglob("*") if p.is_file() and is_image(p)]

            for img_path in images:
                if progress_callback and processed_images % 10 == 0:
                    percent = 10 + int(processed_images / max(1, total_images) * 70)
                    progress_callback(f"📷 Анализ {processed_images}/{total_images}: {folder_name}", percent)

                # Загрузить изображение
                img = imread_safe(img_path)
                if img is None:
                    continue

                # Найти лица
                faces = self.embedder.extract(img)
                if not faces:
                    continue

                # Валидация качества лиц и добавление в глобальный список
                for face_idx, face in enumerate(faces):
                    # Проверка качества лица
                    quality_score = self._validate_face_quality(face, img)

                    # Добавляем только лица достаточного качества
                    if quality_score >= self.quality_threshold:
                        global_face = GlobalFace(
                            embedding=face["embedding"],
                            image_path=img_path,
                            folder_name=folder_name,
                            face_idx=face_idx
                        )
                        self.all_faces.append(global_face)
                    else:
                        print(f"⚠️ Пропущено лицо низкого качества (score={quality_score:.2f}) на {img_path}")

                processed_images += 1

        if progress_callback:
            progress_callback(f"✅ Собрано лиц: {len(self.all_faces)} из {len(self.folder_names)} папок", 80)

    def build_global_clusters(self, progress_callback: ProgressCB = None) -> None:
        """
        Глобальная кластеризация всех собранных лиц.
        """
        if not self.all_faces:
            if progress_callback:
                progress_callback("❌ Нет лиц для кластеризации", 100)
            return

        if progress_callback:
            progress_callback("🔗 Глобальная кластеризация HDBSCAN", 85)

        # Подготовить эмбеддинги для кластеризации
        embeddings = np.vstack([face.embedding for face in self.all_faces]).astype(np.float32)

        # Кластеризация
        labels = cluster_embeddings_hdbscan(
            embeddings,
            min_cluster_size=2,  # минимум 2 лица для кластера
            min_samples=None,
        )

        # Группировка по кластерам
        cluster_faces: Dict[int, List[GlobalFace]] = defaultdict(list)
        for face, label in zip(self.all_faces, labels):
            cluster_faces[label].append(face)

        # Создание глобальных кластеров (исключая шум - label = -1)
        cluster_id = 1
        for label, faces in cluster_faces.items():
            if label == -1:  # Пропустить шум
                continue

            # Найти все папки, где есть этот человек
            folders = {face.folder_name for face in faces}

            cluster = GlobalCluster(
                cluster_id=cluster_id,
                faces=faces,
                folders=folders
            )
            self.global_clusters.append(cluster)
            cluster_id += 1

        if progress_callback:
            progress_callback(f"✅ Глобальных кластеров: {len(self.global_clusters)}", 90)

        # Адаптация порогов качества на основе данных
        if progress_callback:
            progress_callback("🎯 Адаптация порогов качества", 92)

        self._adapt_quality_thresholds()

        # Двухэтапная кластеризация для уточнения результатов
        if progress_callback:
            progress_callback("🔄 Двухэтапная кластеризация", 93)

        self._refine_clusters_two_stage()

        # Пост-обработка кластеров для улучшения качества
        if progress_callback:
            progress_callback("🔧 Пост-обработка кластеров", 95)

        self._post_process_clusters()

        if progress_callback:
            progress_callback(f"✅ Финальных кластеров: {len(self.global_clusters)}", 95)

    def _refine_clusters_two_stage(self) -> None:
        """
        Двухэтапная кластеризация для уточнения результатов.

        Этап 1: Анализ существующих кластеров
        Этап 2: Перекластеризация сомнительных случаев
        """
        if len(self.global_clusters) < 2:
            return

        # Этап 1: Найти сомнительные кластеры
        suspicious_clusters = self._identify_suspicious_clusters()

        if not suspicious_clusters:
            return

        print(f"🔄 Найдено {len(suspicious_clusters)} сомнительных кластеров для перекластеризации")

        # Этап 2: Перекластеризация сомнительных кластеров
        refined_clusters = []
        processed_faces = set()

        for cluster in self.global_clusters:
            if cluster in suspicious_clusters:
                # Перекластеризовать сомнительный кластер
                subclusters = self._recluster_suspicious_cluster(cluster)
                refined_clusters.extend(subclusters)
                # Отметить лица как обработанные
                for face in cluster.faces:
                    processed_faces.add(id(face))
            else:
                # Оставить надежный кластер как есть
                refined_clusters.append(cluster)
                for face in cluster.faces:
                    processed_faces.add(id(face))

        # Проверить, что все лица сохранены
        total_faces = sum(len(c.faces) for c in refined_clusters)
        if total_faces != len(self.all_faces):
            print(f"⚠️ Предупреждение: потеряно {len(self.all_faces) - total_faces} лиц при перекластеризации")

        self.global_clusters = refined_clusters
        print(f"🔄 После двухэтапной кластеризации: {len(self.global_clusters)} кластеров")

    def _identify_suspicious_clusters(self) -> List[GlobalCluster]:
        """
        Найти кластеры, требующие дополнительной проверки.

        Критерии сомнительности:
        - Очень маленькие кластеры (< 3 лиц)
        - Кластеры с низкой внутренней плотностью
        - Кластеры с большим разбросом embeddings
        """
        suspicious = []

        for cluster in self.global_clusters:
            reasons = []

            # Критерий 1: Размер кластера
            if len(cluster.faces) < 3:
                reasons.append("маленький размер")

            # Критерий 2: Разброс embeddings (стандартное отклонение)
            if len(cluster.faces) >= 3:
                embeddings = np.array([face.embedding for face in cluster.faces])
                centroid = np.mean(embeddings, axis=0)
                distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
                std_distance = np.std(distances)

                # Если стандартное отклонение расстояний слишком большое
                if std_distance > 0.5:  # Порог можно настроить
                    reasons.append(f"большой разброс (std={std_distance:.3f})")

            # Критерий 3: Низкая плотность (среднее расстояние до центроида)
            if len(cluster.faces) >= 2:
                avg_distance = np.mean(distances)
                if avg_distance > 0.3:  # Порог можно настроить
                    reasons.append(f"низкая плотность (avg_dist={avg_distance:.3f})")

            if reasons:
                suspicious.append(cluster)
                print(f"🔍 Кластер {cluster.cluster_id} сомнительный: {', '.join(reasons)}")

        return suspicious

    def _recluster_suspicious_cluster(self, cluster: GlobalCluster) -> List[GlobalCluster]:
        """
        Перекластеризовать сомнительный кластер с более строгими параметрами.
        """
        if len(cluster.faces) < 2:
            return [cluster]  # Нельзя перекластеризовать одиночное лицо

        # Извлечь embeddings из кластера
        embeddings = np.array([face.embedding for face in cluster.faces])

        # Более строгие параметры кластеризации
        strict_min_cluster_size = max(2, len(cluster.faces) // 3)  # Минимум 2 или 1/3 от размера
        strict_min_samples = strict_min_cluster_size

        # Перекластеризация
        labels = cluster_embeddings_hdbscan(
            embeddings,
            min_cluster_size=strict_min_cluster_size,
            min_samples=strict_min_samples,
        )

        # Создать новые субкластеры
        subclusters = []
        unique_labels = set(labels.tolist()) - {-1}  # Исключая шум

        for label in unique_labels:
            # Найти лица, принадлежащие этому субкластеру
            subcluster_faces = [
                cluster.faces[i] for i, lbl in enumerate(labels)
                if lbl == label
            ]

            if len(subcluster_faces) >= 2:  # Минимум 2 лица для субкластера
                # Создать новый кластер
                subcluster = GlobalCluster(
                    cluster_id=0,  # ID будет присвоен позже
                    faces=subcluster_faces,
                    folders=cluster.folders  # Сохранить папки
                )
                subclusters.append(subcluster)

        # Если не удалось разделить кластер, вернуть оригинал
        if not subclusters:
            return [cluster]

        print(f"🔄 Кластер {cluster.cluster_id} разделен на {len(subclusters)} субкластеров")
        return subclusters

    def _post_process_clusters(self) -> None:
        """
        Продвинутая пост-обработка кластеров для максимального улучшения качества.

        Выполняет последовательность операций:
        1. Анализ качества кластеров
        2. Удаление мусорных кластеров
        3. Объединение близких кластеров
        4. Разделение смешанных кластеров
        5. Балансировка размеров кластеров
        6. Финальная очистка и пересортировка
        """
        initial_count = len(self.global_clusters)
        print(f"🔧 Начало пост-обработки: {initial_count} кластеров")

        # 1. Анализ и статистика кластеров
        self._analyze_cluster_quality()

        # 2. Удаление мусорных кластеров
        self._remove_noise_clusters()

        # 3. Объединение близких кластеров
        self._merge_similar_clusters()

        # 4. Разделение кластеров с аномалиями
        self._split_anomalous_clusters()

        # 5. Балансировка размеров кластеров
        self._balance_cluster_sizes()

        # 6. Финальная очистка
        self._final_cleanup()

        # 7. Пересортировка ID кластеров
        self._renumber_clusters()

        final_count = len(self.global_clusters)
        print(f"🔧 Пост-обработка завершена: {initial_count} → {final_count} кластеров")

    def _analyze_cluster_quality(self) -> None:
        """
        Проанализировать качество всех кластеров и вывести статистику.
        """
        if not self.global_clusters:
            return

        cluster_sizes = [len(c.faces) for c in self.global_clusters]
        total_faces = sum(cluster_sizes)

        print(f"📊 Анализ кластеров:")
        print(f"   Всего кластеров: {len(self.global_clusters)}")
        print(f"   Всего лиц: {total_faces}")
        print(f"   Средний размер кластера: {total_faces / len(self.global_clusters):.1f}")
        print(f"   Минимальный размер: {min(cluster_sizes)}")
        print(f"   Максимальный размер: {max(cluster_sizes)}")

        # Распределение по размерам
        size_distribution = {}
        for size in cluster_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1

        print("   Распределение по размерам:")
        for size in sorted(size_distribution.keys()):
            count = size_distribution[size]
            print(f"     {size} лиц: {count} кластеров")

        # Проверка на потенциальные проблемы
        tiny_clusters = sum(1 for s in cluster_sizes if s <= 2)
        if tiny_clusters > 0:
            print(f"   ⚠️  Найдено {tiny_clusters} очень маленьких кластеров (≤2 лица)")

        large_clusters = sum(1 for s in cluster_sizes if s >= 20)
        if large_clusters > 0:
            print(f"   ⚠️  Найдено {large_clusters} очень больших кластеров (≥20 лиц)")

    def _split_anomalous_clusters(self) -> None:
        """
        Разделить кластеры с аномальными характеристиками.

        Критерии для разделения:
        - Кластеры с очень большим разбросом embeddings
        - Кластеры с несколькими четко выраженными подгруппами
        """
        clusters_to_split = []

        for cluster in self.global_clusters:
            if len(cluster.faces) < 5:
                continue  # Слишком маленький для анализа

            embeddings = np.array([face.embedding for face in cluster.faces])

            # Проверка на наличие четких подгрупп
            if self._has_clear_subgroups(embeddings):
                clusters_to_split.append(cluster)
                print(f"✂️  Кластер {cluster.cluster_id} имеет четкие подгруппы - будет разделен")

        # Разделить найденные кластеры
        for cluster in clusters_to_split[:]:  # Копия списка
            if cluster in self.global_clusters:  # Проверка, что кластер еще существует
                subclusters = self._split_cluster_by_subgroups(cluster)
                if subclusters and len(subclusters) > 1:
                    # Заменить оригинальный кластер на субкластеры
                    idx = self.global_clusters.index(cluster)
                    self.global_clusters.pop(idx)
                    self.global_clusters.extend(subclusters)
                    print(f"✂️  Кластер {cluster.cluster_id} разделен на {len(subclusters)} частей")

    def _has_clear_subgroups(self, embeddings: np.ndarray, threshold: float = 0.6) -> bool:
        """
        Проверить, есть ли в кластере четко выраженные подгруппы.

        Использует анализ расстояний между точками.
        """
        if len(embeddings) < 6:
            return False

        # Вычислить матрицу расстояний
        distances = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances[i, j] = distances[j, i] = dist

        # Найти среднее расстояние
        avg_distance = np.mean(distances)

        # Проверить, есть ли точки, которые сильно удалены от основной группы
        max_distances = np.max(distances, axis=1)
        outliers = sum(1 for d in max_distances if d > avg_distance * threshold)

        return outliers >= 2  # Минимум 2 выброса для разделения

    def _split_cluster_by_subgroups(self, cluster: GlobalCluster) -> List[GlobalCluster]:
        """
        Разделить кластер на подгруппы с помощью кластеризации.
        """
        if len(cluster.faces) < 4:
            return [cluster]

        embeddings = np.array([face.embedding for face in cluster.faces])

        # Использовать более агрессивную кластеризацию для разделения
        min_cluster_size = max(2, len(cluster.faces) // 4)

        labels = cluster_embeddings_hdbscan(
            embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size // 2
        )

        # Создать субкластеры
        subclusters = []
        unique_labels = set(labels.tolist()) - {-1}

        for label in unique_labels:
            if label == -1:
                continue

            sub_faces = [
                cluster.faces[i] for i, lbl in enumerate(labels)
                if lbl == label
            ]

            if len(sub_faces) >= 2:
                subcluster = GlobalCluster(
                    cluster_id=0,  # ID будет присвоен позже
                    faces=sub_faces,
                    folders=cluster.folders
                )
                subclusters.append(subcluster)

        return subclusters if subclusters else [cluster]

    def _balance_cluster_sizes(self) -> None:
        """
        Балансировка размеров кластеров.

        Если есть очень маленький кластер рядом с большим,
        проверить возможность их объединения с пониженным порогом.
        """
        if len(self.global_clusters) < 2:
            return

        # Найти очень маленькие кластеры
        tiny_clusters = [c for c in self.global_clusters if len(c.faces) <= 3]

        merged_count = 0
        for tiny_cluster in tiny_clusters[:]:  # Копия списка
            if tiny_cluster not in self.global_clusters:
                continue

            # Найти ближайший кластер
            best_match = None
            best_similarity = 0.0

            tiny_centroid = np.mean([face.embedding for face in tiny_cluster.faces], axis=0)

            for cluster in self.global_clusters:
                if cluster == tiny_cluster:
                    continue

                cluster_centroid = np.mean([face.embedding for face in cluster.faces], axis=0)
                similarity = np.dot(tiny_centroid, cluster_centroid) / (
                    np.linalg.norm(tiny_centroid) * np.linalg.norm(cluster_centroid)
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cluster

            # Если найдено хорошее совпадение, объединить
            if best_match and best_similarity > 0.7:  # Пониженный порог для балансировки
                self._merge_two_clusters(tiny_cluster, best_match)
                merged_count += 1
                print(f"⚖️  Маленький кластер объединен с большим (сходство: {best_similarity:.3f})")

        if merged_count > 0:
            print(f"⚖️  Балансировка завершена: объединено {merged_count} кластеров")

    def _merge_two_clusters(self, cluster1: GlobalCluster, cluster2: GlobalCluster) -> None:
        """
        Объединить два кластера.
        """
        # Добавить лица из cluster1 в cluster2
        cluster2.faces.extend(cluster1.faces)

        # Обновить папки
        cluster2.folders.update(cluster1.folders)

        # Удалить cluster1
        if cluster1 in self.global_clusters:
            self.global_clusters.remove(cluster1)

    def _final_cleanup(self) -> None:
        """
        Финальная очистка и валидация кластеров.
        """
        # Удалить пустые кластеры (на всякий случай)
        self.global_clusters = [c for c in self.global_clusters if c.faces]

        # Проверить целостность данных
        total_faces = sum(len(c.faces) for c in self.global_clusters)
        if total_faces != len(self.all_faces):
            print(f"⚠️  Предупреждение: после пост-обработки {total_faces} лиц вместо {len(self.all_faces)}")

        # Проверить, что все лица принадлежат кластерам
        clustered_faces = set()
        for cluster in self.global_clusters:
            for face in cluster.faces:
                face_id = id(face)
                if face_id in clustered_faces:
                    print(f"⚠️  Лицо принадлежит нескольким кластерам!")
                clustered_faces.add(face_id)

        if len(clustered_faces) != total_faces:
            print(f"⚠️  Несоответствие: {len(clustered_faces)} уникальных лиц vs {total_faces} в кластерах")

    def _remove_noise_clusters(self) -> None:
        """
        Удалить кластеры с недостаточным количеством лиц.
        """
        min_faces_per_cluster = 2  # Минимум 2 лица для кластера

        filtered_clusters = []
        for cluster in self.global_clusters:
            if len(cluster.faces) >= min_faces_per_cluster:
                filtered_clusters.append(cluster)
            else:
                print(f"🗑️ Удален мусорный кластер с {len(cluster.faces)} лицами")

        self.global_clusters = filtered_clusters

    def _merge_similar_clusters(self) -> None:
        """
        Объединить кластеры, центроиды которых находятся близко друг к другу.
        """
        if len(self.global_clusters) < 2:
            return

        # Вычислить центроиды кластеров
        centroids = []
        for cluster in self.global_clusters:
            embeddings = np.array([face.embedding for face in cluster.faces])
            centroid = np.mean(embeddings, axis=0)
            centroids.append((cluster, centroid))

        # Найти пары кластеров для объединения
        merge_threshold = 0.8  # Косинусное сходство
        to_merge = []

        for i, (cluster1, centroid1) in enumerate(centroids):
            for j, (cluster2, centroid2) in enumerate(centroids[i+1:], i+1):
                # Косинусное сходство
                similarity = np.dot(centroid1, centroid2) / (
                    np.linalg.norm(centroid1) * np.linalg.norm(centroid2)
                )

                if similarity > merge_threshold:
                    to_merge.append((i, j))
                    print(f"🔗 Объединение кластеров (сходство: {similarity:.3f})")

        # Объединить найденные кластеры
        for i, j in reversed(to_merge):  # В обратном порядке, чтобы индексы оставались корректными
            cluster1 = self.global_clusters[i]
            cluster2 = self.global_clusters[j]

            # Объединить лица
            cluster1.faces.extend(cluster2.faces)

            # Объединить папки
            cluster1.folders.update(cluster2.folders)

            # Удалить второй кластер
            del self.global_clusters[j]

    def _renumber_clusters(self) -> None:
        """
        Пересортировать ID кластеров по порядку.
        """
        for idx, cluster in enumerate(self.global_clusters, 1):
            cluster.cluster_id = idx

    def _adapt_quality_thresholds(self) -> None:
        """
        Адаптировать пороги качества на основе анализа данных.
        """
        if len(self.global_clusters) < 2:
            return

        # Анализ количества лиц в кластерах
        cluster_sizes = [len(cluster.faces) for cluster in self.global_clusters]

        # Если много маленьких кластеров, повышаем порог качества
        small_clusters = sum(1 for size in cluster_sizes if size <= 3)
        small_cluster_ratio = small_clusters / len(self.global_clusters)

        if small_cluster_ratio > 0.6:  # >60% маленьких кластеров
            self.quality_threshold = min(self.quality_threshold + 0.1, 0.9)
            print(f"🎯 Повышен порог качества до {self.quality_threshold:.2f} (много маленьких кластеров)")

        # Если кластеры слишком большие, можно немного понизить порог
        elif len(cluster_sizes) < len(self.folder_names) * 0.5:  # Мало кластеров
            self.quality_threshold = max(self.quality_threshold - 0.05, 0.6)
            print(f"🎯 Понижен порог качества до {self.quality_threshold:.2f} (мало кластеров)")

        # Анализ распределения лиц по папкам
        total_faces = sum(cluster_sizes)
        faces_per_folder = total_faces / len(self.folder_names) if self.folder_names else 0

        if faces_per_folder < 10:  # Мало лиц на папку
            self.min_face_size = max(self.min_face_size - 16, 32)  # Меньше требуемый размер
            print(f"🎯 Уменьшен минимальный размер лица до {self.min_face_size}px")

    def distribute_photos_global(self, root_dir: Path, progress_callback: ProgressCB = None) -> Dict[str, int]:
        """
        Распределить фото по глобальным кластерам.

        Логика:
        - Для каждого кластера и каждой папки создать папку кластера
        - Скопировать ВСЕ фото человека во ВСЕ папки, где он есть
        - Удалить оригиналы

        Returns:
            Статистика распределения
        """
        root_dir = Path(root_dir)
        stats = {"copied": 0, "moved": 0, "clusters_created": len(self.global_clusters)}

        if progress_callback:
            progress_callback("📁 Распределение фото по кластерам", 95)

        # Для каждого глобального кластера
        for cluster in self.global_clusters:
            cluster_id = cluster.cluster_id

            # Для каждой папки, где есть этот человек
            for folder_name in cluster.folders:
                target_folder = root_dir / folder_name / str(cluster_id)
                target_folder.mkdir(parents=True, exist_ok=True)

                # Скопировать все фото этого человека в эту папку
                for photo_path in cluster.photo_paths:
                    target_path = target_folder / photo_path.name

                    if not target_path.exists():
                        try:
                            import shutil
                            shutil.copy2(str(photo_path), str(target_path))
                            stats["copied"] += 1
                        except Exception as e:
                            print(f"⚠️ Ошибка копирования {photo_path}: {e}")

        # Удалить оригиналы после копирования
        if progress_callback:
            progress_callback("🗑️ Очистка оригиналов", 98)

        for cluster in self.global_clusters:
            for photo_path in cluster.photo_paths:
                try:
                    if photo_path.exists():
                        photo_path.unlink()
                        stats["moved"] += 1
                except Exception as e:
                    print(f"⚠️ Ошибка удаления {photo_path}: {e}")

        # Очистить пустые папки
        self._cleanup_empty_folders(root_dir)

        if progress_callback:
            progress_callback(f"✅ Распределено: {stats['copied']} копий, {stats['moved']} перемещений", 100)

        return stats

    def _cleanup_empty_folders(self, root_dir: Path) -> None:
        """Удалить пустые папки после распределения"""
        for folder_name in self.folder_names:
            folder_path = root_dir / folder_name
            if folder_path.exists():
                # Удалить пустые подпапки
                for subfolder in folder_path.iterdir():
                    if subfolder.is_dir():
                        try:
                            subfolder.rmdir()  # Только если пустая
                        except:
                            pass  # Не пустая, пропустить

    def get_cluster_info(self) -> Dict:
        """
        Получить информацию о кластерах для отладки/API.
        """
        return {
            "total_clusters": len(self.global_clusters),
            "total_faces": len(self.all_faces),
            "folders": list(self.folder_names),
            "clusters": [
                {
                    "id": cluster.cluster_id,
                    "face_count": len(cluster.faces),
                    "photo_count": len(cluster.photo_paths),
                    "folders": list(cluster.folders)
                }
                for cluster in self.global_clusters
            ]
        }

# ------------------------
# Вспомогательные функции
# ------------------------

def process_group_global(root_dir: Path, progress_callback: ProgressCB = None,
                        ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)) -> Dict[str, int]:
    """
    Глобальная обработка группы папок.

    Это основная функция, заменяющая process_group_folder.
    """
    # Создать глобальный кластеризатор
    config = ArcFaceConfig(ctx_id=ctx_id, det_size=det_size)
    clusterer = GlobalFaceCluster(config)

    # Шаг 1: Собрать все лица
    clusterer.collect_faces_from_folders(root_dir, progress_callback)

    # Шаг 2: Глобальная кластеризация
    clusterer.build_global_clusters(progress_callback)

    # Шаг 3: Распределение фото
    stats = clusterer.distribute_photos_global(root_dir, progress_callback)

    return stats

# ------------------------
# CLI интерфейс
# ------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Global face clustering")
    parser.add_argument("input", type=str, help="Папка с группой подпапок")
    parser.add_argument("--cpu", action="store_true", help="Принудительно CPU")
    parser.add_argument("--det", type=int, nargs=2, default=[640, 640], help="Размер детектора WxH")

    args = parser.parse_args()

    def cb(msg: str, p: int):
        print(f"[{p:3d}%] {msg}")

    print("🚀 Глобальная кластеризация...")
    stats = process_group_global(
        Path(args.input),
        progress_callback=cb,
        ctx_id=(-1 if args.cpu else 0),
        det_size=tuple(args.det),
    )
    print(f"✅ Готово: {stats}")
