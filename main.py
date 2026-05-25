import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from skan import Skeleton
from load_test_matrices import load_matrices_from_json


def get_skeleton_data(img):
    image_thin = img > 0

    sk = Skeleton(image_thin, keep_images=False)

    graph = sk.graph
    coords = sk.coordinates

    rows, cols = graph.nonzero()

    # Убираем дубли рёбер, потому что graph симметричный
    mask = rows < cols
    rows = rows[mask]
    cols = cols[mask]

    degrees = np.asarray(graph.astype(bool).sum(axis=1)).ravel()

    return {
        "graph": graph,
        "coords": coords,
        "rows": rows,
        "cols": cols,
        "degrees": degrees,
    }


def draw_skeleton(ax, img, skeleton_data, title):
    ax.clear()

    graph = skeleton_data["graph"]
    coords = skeleton_data["coords"]
    rows = skeleton_data["rows"]
    cols = skeleton_data["cols"]
    degrees = skeleton_data["degrees"]

    ax.imshow(img, cmap="gray", interpolation="nearest", origin="upper")

    # Рисуем рёбра графа
    for i, j in zip(rows, cols):
        r0, c0 = coords[i]
        r1, c1 = coords[j]

        ax.plot(
            [c0, c1],
            [r0, r1],
            color="blue",
            linewidth=2,
        )

    # Рисуем узлы графа
    ax.scatter(
        coords[:, 1],
        coords[:, 0],
        s=50,
        color="cyan",
        edgecolors="blue",
        zorder=3,
    )

    ax.set_title(
        f"{title}\n"
        f"nodes={graph.shape[0]}, edges={len(rows)}, degrees={degrees.tolist()}"
    )

    ax.set_aspect("equal")
    ax.set_xticks(range(img.shape[1]))
    ax.set_yticks(range(img.shape[0]))
    ax.grid(color="lightgray", linewidth=0.5)

    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)


def main():
    path = "test_matrices.json"

    matrices = load_matrices_from_json(path)

    skeletons = []

    for img in matrices:
        skeletons.append(get_skeleton_data(img))

    fig, ax = plt.subplots(figsize=(7, 7))

    # Оставляем место снизу под slider
    plt.subplots_adjust(bottom=0.18)

    current_index = 0

    draw_skeleton(
        ax=ax,
        img=matrices[current_index],
        skeleton_data=skeletons[current_index],
        title=f"matrix_{current_index + 1}",
    )

    # Ось для slider
    slider_ax = fig.add_axes([0.2, 0.06, 0.6, 0.04])

    slider = Slider(
        ax=slider_ax,
        label="matrix",
        valmin=0,
        valmax=len(matrices) - 1,
        valinit=0,
        valstep=1,
    )

    def update(value):
        idx = int(slider.val)

        draw_skeleton(
            ax=ax,
            img=matrices[idx],
            skeleton_data=skeletons[idx],
            title=f"matrix_{idx + 1}",
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()