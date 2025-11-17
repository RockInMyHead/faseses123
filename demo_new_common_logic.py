#!/usr/bin/env python3
"""Демонстрация новой логики обработки общих фотографий"""

def demo_old_logic():
    """Старая логика"""
    print("OLD LOGIC (before changes):")
    print("-" * 40)

    # Имитация найденных кластеров
    all_unique_clusters = {1, 3, 5, 7}

    print(f"Found clusters in common photos: {sorted(all_unique_clusters)}")

    created_folders = []
    # Старая логика: перенумерация кластеров
    for i, cluster_id in enumerate(sorted(all_unique_clusters), 1):
        folder_name = str(i)  # 1, 2, 3, 4 вместо 1, 3, 5, 7
        created_folders.append(folder_name)
        print(f"Created folder: {folder_name}/ (was cluster {cluster_id})")

    print(f"Total folders: {len(created_folders)}")
    print(f"Created folders: {created_folders}")
    print()

def demo_new_logic():
    """Новая логика"""
    print("NEW LOGIC (after changes):")
    print("-" * 40)

    # Имитация найденных кластеров
    all_unique_clusters = {1, 3, 5, 7}

    print(f"Found clusters in common photos: {sorted(all_unique_clusters)}")

    created_folders = []

    # Новая логика: сохраняем реальные номера кластеров
    for cluster_id in sorted(all_unique_clusters):
        folder_name = str(cluster_id)  # Сохраняем 1, 3, 5, 7
        created_folders.append(folder_name)
        print(f"Created folder: {folder_name}/ (cluster {cluster_id})")

    # Добавляем 2 дополнительные пустые папки
    max_cluster_id = max(all_unique_clusters) if all_unique_clusters else 0
    for i in range(1, 3):  # Создаем 2 дополнительные папки
        extra_cluster_id = max_cluster_id + i
        folder_name = str(extra_cluster_id)
        created_folders.append(folder_name)
        print(f"Created extra empty folder: {folder_name}/")

    print(f"Total folders: {len(created_folders)}")
    print(f"Created folders: {created_folders}")
    print()

def main():
    print("COMMON PHOTOS LOGIC - BEFORE vs AFTER")
    print("=" * 60)
    print()

    demo_old_logic()
    demo_new_logic()

    print("SUMMARY OF CHANGES:")
    print("=" * 60)
    print("✓ Real cluster numbers preserved (no renumbering)")
    print("✓ Folders created for EACH cluster from common photos")
    print("✓ Empty folders created even if no individual photos exist")
    print("✓ 2 additional empty folders always added")
    print("✓ Better organization and consistency")

    print()
    print("BENEFITS:")
    print("- More predictable folder structure")
    print("- Easier to match common photos with individual folders")
    print("- Space for future photos of the same people")
    print("- Consistent numbering across the album")

if __name__ == "__main__":
    main()
