from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from vectorization_skan.get_skeleton_data import get_skeleton_data as skan_vectorization
from pyskeletvectorization import vectorize as neighborhood_vectorization


def main():
    image_path = Path("skelettest/skeleton_1.png")

    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    # Векторизация выполняется ДО crop — на полном изображении
    start_time = time.perf_counter()
    lines = neighborhood_vectorization(img_gray)
    elapsed_time = time.perf_counter() - start_time

    print(f"Время векторизации: {elapsed_time:.6f} сек")
    print(f"Всего линий после векторизации: {len(lines)}")

    # Crop только для отображения
    cropped, crop_x, crop_y, crop_size = center_crop_1024(img_gray)

    # Только фильтрация линий, без нарезки на сегменты
    visible_lines = prepare_lines_for_crop(
        lines,
        crop_x,
        crop_y,
        crop_size
    )

    print(f"Линий, касающихся crop: {len(visible_lines)}")

    # Узлы линий в координатах crop
    node_points = collect_line_nodes(visible_lines)

    print(f"Узлов линий в crop: {len(node_points)}")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(cropped, cmap="gray", zorder=0)
    ax.axis("off")

    # Черные точки узлов рисуются ПОД линиями
    if len(node_points) > 0:
        ax.scatter(
            node_points[:, 0],
            node_points[:, 1],
            s=8,
            c="blue",
            marker="o",
            linewidths=0,
            zorder=1
        )

    # Быстрее, чем делать ax.plot(...) для каждой линии отдельно
    if visible_lines:
        colors = np.random.rand(len(visible_lines), 3)

        line_collection = LineCollection(
            visible_lines,
            colors=colors,
            linewidths=1,
            zorder=2
        )

        ax.add_collection(line_collection)

    # Показываем только область crop.
    # Линии и точки могут выходить за края, matplotlib сам их визуально обрежет.
    ax.set_xlim(0, crop_size)
    ax.set_ylim(crop_size, 0)

    plt.show()


def center_crop_1024(img):
    h, w = img.shape[:2]
    crop_size = 1024

    if h < crop_size or w < crop_size:
        raise ValueError(f"Изображение меньше 1024x1024: {w}x{h}")

    crop_y = (h - crop_size) // 2
    crop_x = (w - crop_size) // 2

    cropped = img[
        crop_y:crop_y + crop_size,
        crop_x:crop_x + crop_size
    ]

    return cropped, crop_x, crop_y, crop_size


def line_bbox_touches_crop(line, crop_x, crop_y, crop_size):
    """
    Быстрая проверка: пересекается ли bbox линии с bbox crop.
    Линию не режем и не сплитим.
    """

    line = np.asarray(line)

    crop_x2 = crop_x + crop_size
    crop_y2 = crop_y + crop_size

    # Формат: [[x1, y1], [x2, y2], ...]
    if line.ndim == 2 and line.shape[1] == 2:
        if len(line) == 0:
            return False

        xs = line[:, 0]
        ys = line[:, 1]

        line_x1 = np.min(xs)
        line_x2 = np.max(xs)
        line_y1 = np.min(ys)
        line_y2 = np.max(ys)

    # Формат: [x1, y1, x2, y2]
    elif line.size == 4:
        x1, y1, x2, y2 = line.ravel()

        line_x1 = min(x1, x2)
        line_x2 = max(x1, x2)
        line_y1 = min(y1, y2)
        line_y2 = max(y1, y2)

    else:
        return False

    # Если bbox линии полностью левее/правее/выше/ниже crop — выбрасываем
    if line_x2 < crop_x:
        return False

    if line_x1 > crop_x2:
        return False

    if line_y2 < crop_y:
        return False

    if line_y1 > crop_y2:
        return False

    return True


def prepare_lines_for_crop(lines, crop_x, crop_y, crop_size):
    """
    Выбрасывает линии, bbox которых не касается crop.
    Остальные линии не режет, только переводит координаты
    из системы полного изображения в систему crop.
    """

    visible_lines = []

    for line in lines:
        line = np.asarray(line)

        if not line_bbox_touches_crop(line, crop_x, crop_y, crop_size):
            continue

        # Формат: [[x1, y1], [x2, y2], ...]
        if line.ndim == 2 and line.shape[1] == 2:
            if len(line) < 2:
                continue

            local_line = line.astype(np.float32).copy()
            local_line[:, 0] -= crop_x
            local_line[:, 1] -= crop_y

            visible_lines.append(local_line)

        # Формат: [x1, y1, x2, y2]
        elif line.size == 4:
            x1, y1, x2, y2 = line.ravel()

            local_line = np.array(
                [
                    [x1 - crop_x, y1 - crop_y],
                    [x2 - crop_x, y2 - crop_y]
                ],
                dtype=np.float32
            )

            visible_lines.append(local_line)

    return visible_lines


def collect_line_nodes(lines):
    """
    Собирает точки узлов линий.

    Для полилиний формата [[x1, y1], [x2, y2], ...]
    узлами считаются все точки полилинии.

    Для отрезков формата [[x1, y1], [x2, y2]]
    узлами считаются обе конечные точки.
    """

    if not lines:
        return np.empty((0, 2), dtype=np.float32)

    nodes = []

    for line in lines:
        line = np.asarray(line, dtype=np.float32)

        if line.ndim == 2 and line.shape[1] == 2 and len(line) > 0:
            nodes.append(line)

    if not nodes:
        return np.empty((0, 2), dtype=np.float32)

    nodes = np.vstack(nodes)

    # Убираем дубликаты узлов, чтобы не рисовать одну и ту же точку много раз
    nodes = np.unique(nodes, axis=0)

    return nodes


if __name__ == "__main__":
    main()