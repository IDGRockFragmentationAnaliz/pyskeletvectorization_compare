import numpy as np
import trace_skeleton


def get_skeleton_data(skel: np.ndarray) -> list[np.ndarray]:
    """
    Input:
        skel[y, x] == 0/1

    Output:
        [
            np.array([[x1, y1], [x2, y2], ...], dtype=np.int32),
            ...
        ]
    """
    if skel.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={skel.shape}")

    img = (skel > 0).astype(np.uint8)

    raw_lines = trace_skeleton.from_numpy(
        img,
        csize=10,
        maxIter=999,
    )

    lines = [
        np.asarray(line, dtype=np.int32)
        for line in raw_lines
        if len(line) >= 2
    ]

    return lines