"""
Альтернативная версия кластеризации с использованием face_recognition (dlib-based).
Более стабильная, чем insightface, но менее точная для больших датасетов.

Зависимости:
    pip install face-recognition==1.3.0
    pip install face-recognition-models==0.3.0
    pip install dlib==19.24.6
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from collections import defaultdict, deque

# Faiss может отсутствовать при сборке — валидируем импорт.
try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None

try:
    import face_recognition
except Exception as e:  # pragma: no cover
    face_recognition = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
ProgressCB = Optional[Callable[[str, int], None]]

# ------------------------
# Утилиты ввода/вывода
# ------------------------

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def imread_safe(path: Path) -> Optional[np.ndarray]:
    """Аккуратное чтение изображений для face_recognition."""
    try:
        # face_recognition работает с PIL или numpy массивами
        image = face_recognition.load_image_file(str(path))
        return image
    except Exception:
        return None

# ------------------------
# Инициализация модели face_recognition
# ------------------------
@dataclass
class FaceRecConfig:
    model: str = "cnn"  # "hog" или "cnn" (cnn точнее но медленнее)
    tolerance: float = 0.5  # порог для сравнения лиц (меньше = строже)


class FaceRecEmbedder:
    def __init__(self, config: FaceRecConfig = FaceRecConfig()):
        if face_recognition is None:
            raise ImportError("face_recognition не установлен. Установите пакет face_recognition.")
        self.config = config

    def extract(self, img_rgb: np.ndarray) -> List[Dict]:
        """Возвращает список лиц: [{encoding, location}]."""
        # face_recognition.load_image_file уже возвращает RGB
        face_locations = face_recognition.face_locations(img_rgb, model=self.config.model)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        results = []
        for location, encoding in zip(face_locations, face_encodings):
            results.append({
                "encoding": encoding,  # 128-dimensional encoding
                "location": location,  # (top, right, bottom, left)
            })
        return results

# ------------------------
# Кластеризация через Faiss (или простой метод)
# ------------------------
@dataclass
class ClusterParams:
    sim_threshold: float = 0.45   # ниже порог для face_recognition (строже)
    min_cluster_size: int = 2     # срезаем мелкие компоненты как одиночки
    use_faiss: bool = True        # использовать Faiss или простой метод


def _build_similarity_graph_faiss(embeddings: np.ndarray, params: ClusterParams) -> List[List[int]]:
    if faiss is None:
        raise ImportError("faiss не установлен. Установите faiss-gpu или faiss-cpu.")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Эмбеддинги уже L2-нормированы для cosine=dot
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # range_search: вернёт пары (i,j) с sim >= threshold
    lims, D, I = index.range_search(embeddings, params.sim_threshold)

    # Формируем списки смежности
    n = embeddings.shape[0]
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        beg, end = lims[i], lims[i + 1]
        pairs = sorted(zip(I[beg:end], D[beg:end]), key=lambda t: -t[1])
        for j, sim in pairs:
            if j == i or j < 0:
                continue
            adj[i].append(int(j))
    return adj


def _simple_clustering(embeddings: np.ndarray, params: ClusterParams) -> np.ndarray:
    """Простой метод кластеризации без Faiss"""
    n = len(embeddings)
    labels = -np.ones(n, dtype=np.int32)
    cid = 0

    for i in range(n):
        if labels[i] != -1:
            continue

        # Находим все похожие лица для этого лица
        similar = [i]
        for j in range(n):
            if i == j or labels[j] != -1:
                continue
            # Косинусная близость
            sim = np.dot(embeddings[i], embeddings[j])
            if sim >= params.sim_threshold:
                similar.append(j)

        if len(similar) >= params.min_cluster_size:
            for idx in similar:
                labels[idx] = cid
            cid += 1

    return labels


def _connected_components(adj: List[List[int]]) -> np.ndarray:
    n = len(adj)
    labels = -np.ones(n, dtype=np.int32)
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        q = deque([i])
        labels[i] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = cid
                    q.append(v)
        cid += 1
    return labels


def cluster_embeddings(embeddings: np.ndarray, params: ClusterParams) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([], dtype=np.int32)

    if params.use_faiss and faiss is not None:
        adj = _build_similarity_graph_faiss(embeddings, params)
        labels = _connected_components(adj)
    else:
        labels = _simple_clustering(embeddings, params)

    # Отфильтруем мелкие кластеры
    sizes = defaultdict(int)
    for lb in labels:
        sizes[int(lb)] += 1

    for i, lb in enumerate(labels):
        if lb != -1 and sizes[int(lb)] < params.min_cluster_size:
            labels[i] = -1

    # Сжимаем индексы кластеров
    if np.any(labels != -1):
        uniq = sorted(x for x in set(labels.tolist()) if x != -1)
        remap = {old: i for i, old in enumerate(uniq)}
        out = labels.copy()
        for i, lb in enumerate(labels):
            if lb == -1:
                out[i] = -1
            else:
                out[i] = remap[int(lb)]
        return out
    return labels

# ------------------------
# Основной пайплайн
# ------------------------

def build_plan_face_recognition(
    input_dir: Path,
    progress_callback: ProgressCB = None,
    sim_threshold: float = 0.6,
    min_cluster_size: int = 2,
    model: str = "hog",
) -> Dict:
    """Кластеризация лиц с face_recognition + Faiss.

    Возвращает dict аналогичный build_plan_pro
    """
    t0 = time.time()
    input_dir = Path(input_dir)
    if progress_callback:
        progress_callback(f"🚀 Кластеризация: {input_dir}", 2)

    # Инициализация эмбеддера
    emb = FaceRecEmbedder(FaceRecConfig(model=model, tolerance=sim_threshold))

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
        if progress_callback and (i % 5 == 0):  # реже обновляем прогресс
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

        img_face_count[img_path] = len(faces)
        for face in faces:
            all_embeddings.append(face["encoding"])
            owners.append(img_path)

    if not all_embeddings:
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
        }

    X = np.vstack(all_embeddings).astype(np.float32)

    # Кластеризация
    if progress_callback:
        progress_callback("🔗 Построение графа похожести", 70)
    labels = cluster_embeddings(
        X,
        ClusterParams(sim_threshold=sim_threshold, min_cluster_size=min_cluster_size),
    )

    if progress_callback:
        progress_callback(f"✅ Кластеров: {len(set(labels.tolist()) - {-1})}", 85)

    # Формирование мапов
    cluster_map: Dict[int, set[Path]] = defaultdict(set)
    cluster_by_img: Dict[Path, set[int]] = defaultdict(set)

    for lb, path in zip(labels, owners):
        if lb == -1:
            continue
        cluster_map[int(lb)].add(path)
        cluster_by_img[path].add(int(lb))

    # План перемещений
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

EXCLUDED_COMMON_NAMES = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]


def distribute_to_folders(plan: dict, base_dir: Path, cluster_start: int = 1, progress_callback: ProgressCB = None, common_mode: bool = False) -> Tuple[int, int, int]:
    import shutil

    moved, copied = 0, 0
    moved_paths = set()

    used_clusters = sorted({c for item in plan.get("plan", []) for c in item["cluster"]})
    common_photo_clusters = set()
    if common_mode:
        for item in plan.get("plan", []):
            src = Path(item["path"])
            is_common_photo = any(excluded_name in str(src.parent).lower() for excluded_name in EXCLUDED_COMMON_NAMES)
            if is_common_photo:
                common_photo_clusters.update(item["cluster"])

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
        is_common_photo = any(excluded_name in str(src.parent).lower() for excluded_name in EXCLUDED_COMMON_NAMES)
        if not is_common_photo:
            clusters = [cluster_id_map[c] for c in item["cluster"]]
            for cid in clusters:
                cluster_file_counts[cid] = cluster_file_counts.get(cid, 0) + 1

    for i, item in enumerate(plan_items):
        if progress_callback:
            percent = int((i + 1) / max(total_items, 1) * 100)
            progress_callback(f"📁 Распределение файлов: {percent}% ({i+1}/{total_items})", percent)

        src = Path(item["path"])
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        if not src.exists():
            continue

        is_common_photo = any(excluded_name in str(src.parent).lower() for excluded_name in EXCLUDED_COMMON_NAMES)

        if is_common_photo:
            print(f"📌 Общая фотография оставлена: {src.name}")
            continue

        if len(clusters) == 1:
            parent_folder = src.parent
            dst = parent_folder / f"{clusters[0]}" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve():
                shutil.move(str(src), str(dst))
                moved += 1
                moved_paths.add(src.parent)
        else:
            parent_folder = src.parent
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
                real_count = 0
                for file_path in folder_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        real_count += 1

                if real_count == 0:
                    try:
                        folder_path.rmdir()
                        print(f"🗑️ Удалена пустая папка: {folder_path}")
                    except Exception:
                        pass

    # Очистка пустых каталогов
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

    return moved, copied, cluster_start + len(used_clusters)


# ------------------------
# Групповая обработка
# ------------------------

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
                                   sim_threshold: float = 0.6, min_cluster_size: int = 2,
                                   model: str = "hog") -> Tuple[int, int]:
    data = build_plan_face_recognition(common_dir, progress_callback=progress_callback,
                                     sim_threshold=sim_threshold, min_cluster_size=min_cluster_size,
                                     model=model)
    moved, copied, _ = distribute_to_folders(data, common_dir, cluster_start=1, progress_callback=progress_callback, common_mode=True)
    return moved, copied


def process_group_folder(group_dir: Path, progress_callback: ProgressCB = None,
                         include_excluded: bool = False,
                         sim_threshold: float = 0.6, min_cluster_size: int = 2,
                         model: str = "hog", det_size: Tuple[int, int] = (640, 640)) -> Tuple[int, int, int]:
    group_dir = Path(group_dir)

    if include_excluded:
        commons = find_common_folders_recursive(group_dir)
        for i, c in enumerate(commons):
            if progress_callback:
                progress_callback(f"📋 Общие: {c.name} ({i+1}/{len(commons)})", 5 + int(i / max(1, len(commons)) * 20))
            process_common_folder_at_level(c, progress_callback=progress_callback,
                                           sim_threshold=sim_threshold, min_cluster_size=min_cluster_size,
                                           model=model)

    subdirs = [d for d in sorted(group_dir.iterdir()) if d.is_dir()]
    if not include_excluded:
        subdirs = [d for d in subdirs if all(ex not in d.name.lower() for ex in EXCLUDED_COMMON_NAMES)]

    total = len(subdirs)
    moved_all, copied_all = 0, 0
    next_cluster_id = 1  # Глобальный счетчик кластеров для всей группы

    for i, sub in enumerate(subdirs):
        if progress_callback:
            progress_callback(f"🔍 {sub.name}: кластеризация ({i+1}/{total})", 25 + int(i / max(1, total) * 60))
        data = build_plan_face_recognition(
            input_dir=sub,
            progress_callback=progress_callback,
            sim_threshold=sim_threshold,
            min_cluster_size=min_cluster_size,
            model=model,
        )
        m, c, next_cluster_id = distribute_to_folders(data, sub, cluster_start=next_cluster_id, progress_callback=progress_callback)
        moved_all += m
        copied_all += c

    return moved_all, copied_all, next_cluster_id


# ------------------------
# CLI-обвязка
# ------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face recognition + Faiss face clustering")
    parser.add_argument("input", type=str, help="Папка с изображениями или группа папок")
    parser.add_argument("--group", action="store_true", help="Обрабатывать как группу подпапок")
    parser.add_argument("--include-common", action="store_true", help="Обрабатывать папки 'общие' внутри группы")
    parser.add_argument("--sim", type=float, default=0.6, help="Порог косинусной близости [0..1]")
    parser.add_argument("--minsz", type=int, default=2, help="Мин. размер кластера")
    parser.add_argument("--model", type=str, default="hog", choices=["hog", "cnn"], help="Модель детектора")

    args = parser.parse_args()

    def cb(msg: str, p: int):
        print(f"[{p:3d}%] {msg}")

    if args.group:
        moved, copied, _ = process_group_folder(
            Path(args.input), progress_callback=cb,
            include_excluded=args.include_common,
            sim_threshold=args.sim, min_cluster_size=args.minsz,
            model=args.model,
        )
        print(f"DONE: moved={moved}, copied={copied}")
    else:
        data = build_plan_face_recognition(
            Path(args.input), progress_callback=cb,
            sim_threshold=args.sim, min_cluster_size=args.minsz,
            model=args.model,
        )
        m, c, _ = distribute_to_folders(data, Path(args.input), cluster_start=1, progress_callback=cb)
        print(f"DONE: moved={m}, copied={c}")
