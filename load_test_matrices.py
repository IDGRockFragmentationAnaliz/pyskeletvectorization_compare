import json
from pathlib import Path
import numpy as np


def load_matrices_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_matrices = json.load(f)

    matrices = []

    for matrix in raw_matrices:
        img = np.array(matrix, dtype=np.uint8) * 255
        matrices.append(img)

    return matrices