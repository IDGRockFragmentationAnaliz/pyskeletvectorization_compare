import numpy as np
import matplotlib.pyplot as plt

from pi2.pi2py2 import *


def normalize_points(points_np: np.ndarray) -> np.ndarray:
    """
    Приводит pi2 points к форме (N, 3).
    getpointsandlines должен возвращать строки как (x, y, z),
    но делаем защитную нормализацию.
    """
    points_np = np.asarray(points_np)
    points_np = np.squeeze(points_np)

    if points_np.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if points_np.ndim == 1:
        if points_np.size % 3 != 0:
            raise ValueError(f"Не могу интерпретировать points shape={points_np.shape}")
        return points_np.reshape(-1, 3).astype(np.float32)

    if points_np.ndim == 2:
        if points_np.shape[1] == 3:
            return points_np.astype(np.float32)
        if points_np.shape[0] == 3:
            return points_np.T.astype(np.float32)

    raise ValueError(f"Не могу интерпретировать points shape={points_np.shape}")


def parse_compressed_lines(lines_np: np.ndarray, points_xyz: np.ndarray) -> list[np.ndarray]:
    flat = np.asarray(lines_np).ravel().astype(np.int64)

    if flat.size == 0:
        return []

    edge_count = int(flat[0])
    offset = 1

    polylines = []

    for edge_i in range(edge_count):
        point_count = int(flat[offset])
        offset += 1

        point_indices = flat[offset:offset + point_count]
        offset += point_count

        line_yx = points_xyz[point_indices][:, :2]

        # ВАЖНО: numpy row,col -> matplotlib x,y
        line_xy = line_yx[:, ::-1]

        polylines.append(line_xy)

    return polylines

def pi2_vectorize_to_polylines(img: np.ndarray):
    pi = Pi2()

    skel_np = (img > 0).astype(np.uint8) * 255

    # tracelineskeleton зануляет skeleton image, поэтому передаем копию.
    skel_pi = pi.newimage(ImageDataType.UINT8)
    skel_pi.from_numpy(skel_np.copy())

    vertices = pi.newimage(ImageDataType.FLOAT32)
    edges = pi.newimage(ImageDataType.UINT64)
    measurements = pi.newimage(ImageDataType.FLOAT32)
    edge_points = pi.newimage(ImageDataType.INT32)

    # store_all_edge_points=True обязательно для getpointsandlines.
    pi.tracelineskeleton(
        skel_pi,
        vertices,
        edges,
        measurements,
        edge_points,
        True,
        1,
    )

    points = pi.newimage(ImageDataType.FLOAT32)
    lines = pi.newimage(ImageDataType.UINT64)

    # Официальная конвертация graph -> points-and-lines.
    pi.getpointsandlines(
        vertices,
        edges,
        measurements,
        edge_points,
        points,
        lines,
    )

    vertices_np = np.asarray(vertices.to_numpy())
    edges_np = np.asarray(edges.to_numpy())
    measurements_np = np.asarray(measurements.to_numpy())
    edge_points_np = np.asarray(edge_points.to_numpy())

    points_np = np.asarray(points.to_numpy())
    lines_np = np.asarray(lines.to_numpy())

    points_xyz = normalize_points(points_np)
    polylines_xy = parse_compressed_lines(lines_np, points_xyz)

    debug = {
        "vertices": vertices_np,
        "edges": edges_np,
        "measurements": measurements_np,
        "edge_points": edge_points_np,
        "points": points_np,
        "lines": lines_np,
        "points_xyz_normalized": points_xyz,
    }

    return polylines_xy, debug


def draw_result(img, polylines, debug):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(img, cmap="gray", interpolation="nearest", origin="upper")

    for i, line in enumerate(polylines):
        if len(line) == 0:
            continue

        ax.plot(
            line[:, 0],
            line[:, 1],
            color="blue",
            linewidth=2,
            zorder=2,
        )

        ax.scatter(
            line[:, 0],
            line[:, 1],
            s=45,
            color="cyan",
            edgecolors="blue",
            zorder=3,
        )

        # Номер линии около первой точки
        ax.text(
            line[0, 0] + 0.05,
            line[0, 1] + 0.05,
            f"L{i}",
            color="blue",
            fontsize=9,
            zorder=5,
        )

    # Рисуем pi2 vertices отдельно.
    vertices = np.squeeze(debug["vertices"])
    if vertices.size > 0:
        if vertices.ndim == 1:
            vertices = vertices.reshape(3, -1)
        elif vertices.shape[0] != 3 and vertices.shape[1] == 3:
            vertices = vertices.T

        vx = vertices[1]
        vy = vertices[0]

        ax.scatter(
            vx,
            vy,
            s=140,
            color="red",
            marker="x",
            linewidths=2,
            zorder=4,
            label="pi2 vertices",
        )

        for i, (x, y) in enumerate(zip(vx, vy)):
            ax.text(
                x + 0.05,
                y + 0.05,
                f"V{i}",
                color="red",
                fontsize=10,
                zorder=5,
            )

    total_points = sum(len(line) for line in polylines)

    ax.set_title(
        "pi2.tracelineskeleton + getpointsandlines\n"
        f"polylines={len(polylines)}, polyline_points={total_points}"
    )

    ax.set_aspect("equal")
    ax.set_xticks(range(img.shape[1]))
    ax.set_yticks(range(img.shape[0]))
    ax.grid(color="lightgray", linewidth=0.5)

    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)

    ax.legend(loc="upper right")
    plt.show()


def print_debug(polylines, debug):
    print("=" * 80)
    print("POLYLINES")
    print(f"Количество линий: {len(polylines)}")

    for i, line in enumerate(polylines):
        print(f"\nline[{i}], shape={line.shape}:")
        print(line)

    print("\n" + "=" * 80)
    print("RAW PI2 OUTPUTS")

    for key, value in debug.items():
        arr = np.asarray(value)
        print(f"\n{key}: shape={arr.shape}, dtype={arr.dtype}")
        print(arr)


def main():
    img = np.array(
        [
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )

    polylines, debug = pi2_vectorize_to_polylines(img)

    print_debug(polylines, debug)
    draw_result(img, polylines, debug)


if __name__ == "__main__":
    main()