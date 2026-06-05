import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patheffects as pe
from skan import Skeleton
from load_test_matrices import load_matrices_from_json
from vectorization_skan.get_skeleton_data import get_skeleton_data as skan_vectorization
from vectorization_lingdong.get_skeleton_data import get_skeleton_data as lindong_vectorization
from vectorization_pi2 import vectorize_skeleton as pi2_vectorization
from pyskeletvectorization import vectorize as neighborhood_vectorization


def main():
    path = "test_matrices.json"

    matrices = load_matrices_from_json(path)

    all_lines = []

    for img in matrices:
        lines = skan_vectorization(img)
        #lines = pi2_vectorization(img)
        #lines = lindong_vectorization(img)
        #lines = neighborhood_vectorization(img)
        all_lines.append(lines)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Оставляем место снизу под slider
    plt.subplots_adjust(bottom=0.18)

    current_index = 0

    draw_skeleton(
        ax=ax,
        img=matrices[current_index],
        lines=all_lines[current_index],
        title=f"matrix_{current_index + 1}",
    )

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
            lines=all_lines[idx],
            title=f"matrix_{idx + 1}",
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def draw_skeleton(ax, img, lines, title):
    ax.clear()

    ax.imshow(img, cmap="gray", interpolation="nearest", origin="upper")

    # Рисуем ломаные линии
    for line in lines:
        if len(line) == 1:
            ax.scatter(
                line[:, 0],
                line[:, 1],
                s=60,
                color="white",
                edgecolors="black",
                zorder=3,
            )
        else:
            ax.plot(
                line[:, 0],
                line[:, 1],
                color="white",
                linewidth=1,
                path_effects=[
                    pe.Stroke(linewidth=3, foreground="black"),
                    pe.Normal(),
                ],
            )

            # Точки ломаной
            ax.scatter(
                line[:, 0],
                line[:, 1],
                s=35,
                color="white",
                edgecolors="black",
                zorder=3,
            )

    total_points = sum(len(line) for line in lines)

    ax.set_title(
        f"{title}\n"
        f"polylines={len(lines)}, points={total_points}"
    )

    ax.set_aspect("equal")

    h, w = img.shape

    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))

    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.grid(which="major", visible=False)

    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    plt.savefig('pictures/1.png', dpi=300)


if __name__ == "__main__":
    main()