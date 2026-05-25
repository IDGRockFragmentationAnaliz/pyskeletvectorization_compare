import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from skan import Skeleton
from skan_vectorization.get_skeleton_data import get_skeleton_data
from load_test_matrices import load_matrices_from_json

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
                color="cyan",
                edgecolors="blue",
                zorder=3,
            )
        else:
            ax.plot(
                line[:, 0],
                line[:, 1],
                color="blue",
                linewidth=2,
            )

            # Точки ломаной
            ax.scatter(
                line[:, 0],
                line[:, 1],
                s=35,
                color="cyan",
                edgecolors="blue",
                zorder=3,
            )

    total_points = sum(len(line) for line in lines)

    ax.set_title(
        f"{title}\n"
        f"polylines={len(lines)}, points={total_points}"
    )

    ax.set_aspect("equal")
    ax.set_xticks(range(img.shape[1]))
    ax.set_yticks(range(img.shape[0]))
    ax.grid(color="lightgray", linewidth=0.5)

    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)


def print_lines(title, lines):
    print("=" * 80)
    print(title)
    print(f"Количество ломаных: {len(lines)}")

    for idx, line in enumerate(lines):
        print(f"line[{idx}]:")
        print(line)


def main():
    path = "test_matrices.json"

    matrices = load_matrices_from_json(path)

    all_lines = []

    for img in matrices:
        lines = get_skeleton_data(img)
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

    print_lines(
        title=f"matrix_{current_index + 1}",
        lines=all_lines[current_index],
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

        print_lines(
            title=f"matrix_{idx + 1}",
            lines=all_lines[idx],
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()