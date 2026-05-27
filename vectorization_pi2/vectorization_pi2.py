import numpy as np
from pi2.pi2py2 import *


def pi2_vectorize_skeleton(img: np.ndarray) -> list[np.ndarray]:
    """
    Vectorize 1-pixel skeleton image using pi2.tracelineskeleton.

    Input:
        img: 2D numpy array.
             Non-zero pixels are treated as skeleton pixels.

    Output:
        [
            np.array([[x1, y1], [x2, y2], ...], dtype=np.int32),
            ...
        ]

    Notes:
        - Coordinates are returned as matplotlib/image coordinates: x=col, y=row.
        - pi2 may create centroid points for junction regions.
          Those centroid coordinates can be fractional; here they are rounded to int32.
    """
    img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={img.shape}")

    pi = Pi2()

    skel_np = (img > 0).astype(np.uint8) * 255

    # tracelineskeleton mutates/clears input image, so use a copy.
    skel_pi = pi.newimage(ImageDataType.UINT8)
    skel_pi.from_numpy(skel_np.copy())

    vertices = pi.newimage(ImageDataType.FLOAT32)
    edges = pi.newimage(ImageDataType.UINT64)
    measurements = pi.newimage(ImageDataType.FLOAT32)
    edge_points = pi.newimage(ImageDataType.INT32)

    pi.tracelineskeleton(
        skel_pi,
        vertices,
        edges,
        measurements,
        edge_points,
        True,   # store all edge points
        1,      # thread count
    )

    points = pi.newimage(ImageDataType.FLOAT32)
    lines = pi.newimage(ImageDataType.UINT64)

    pi.getpointsandlines(
        vertices,
        edges,
        measurements,
        edge_points,
        points,
        lines,
    )

    points_xyz = _normalize_pi2_points(points.to_numpy())
    polylines = _parse_pi2_lines(lines.to_numpy(), points_xyz)

    return polylines


def _normalize_pi2_points(points_np: np.ndarray) -> np.ndarray:
    """
    Normalize pi2 points array to shape (N, 3).

    pi2 point coordinates are treated as array-axis coordinates:
        [row, col, z]
    or similar, depending on pi2 numpy bridge.

    Later we convert:
        row, col -> x=col, y=row
    """
    points_np = np.asarray(points_np)
    points_np = np.squeeze(points_np)

    if points_np.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if points_np.ndim == 1:
        if points_np.size % 3 != 0:
            raise ValueError(f"Cannot interpret pi2 points shape={points_np.shape}")
        return points_np.reshape(-1, 3).astype(np.float32)

    if points_np.ndim == 2:
        if points_np.shape[1] == 3:
            return points_np.astype(np.float32)
        if points_np.shape[0] == 3:
            return points_np.T.astype(np.float32)

    raise ValueError(f"Cannot interpret pi2 points shape={points_np.shape}")


def _parse_pi2_lines(lines_np: np.ndarray, points_xyz: np.ndarray) -> list[np.ndarray]:
    """
    Parse pi2 getpointsandlines() compressed lines format.

    Expected compressed format:
        [line_count,
         point_count_0, idx_0_0, idx_0_1, ...,
         point_count_1, idx_1_0, idx_1_1, ...,
         ...]

    Returns:
        list of np.ndarray, each shape (N, 2), dtype int32.
    """
    flat = np.asarray(lines_np).ravel().astype(np.int64)

    if flat.size == 0:
        return []

    line_count = int(flat[0])
    offset = 1

    polylines: list[np.ndarray] = []

    for line_idx in range(line_count):
        if offset >= flat.size:
            raise ValueError(f"Broken pi2 lines array before line {line_idx}")

        point_count = int(flat[offset])
        offset += 1

        if point_count < 0:
            raise ValueError(f"Negative point count in line {line_idx}: {point_count}")

        if offset + point_count > flat.size:
            raise ValueError(
                f"Broken pi2 lines array at line {line_idx}: "
                f"need {point_count} point indices, "
                f"available {flat.size - offset}"
            )

        point_indices = flat[offset:offset + point_count]
        offset += point_count

        coords = points_xyz[point_indices]

        # pi2/numpy image coordinates are interpreted as row,col.
        # Required output is x,y, so swap first two coordinates.
        xy_float = coords[:, [1, 0]]

        # pi2 junction centroids may be fractional.
        # Required output dtype is int32, so round to nearest pixel coordinate.
        xy_int = np.rint(xy_float).astype(np.int32)

        # Optional cleanup: remove consecutive duplicates after rounding.
        xy_int = _remove_consecutive_duplicate_points(xy_int)

        polylines.append(xy_int)

    return polylines


def _remove_consecutive_duplicate_points(line: np.ndarray) -> np.ndarray:
    """
    Remove consecutive duplicate points caused by centroid rounding.
    """
    if len(line) <= 1:
        return line

    keep = np.ones(len(line), dtype=bool)
    keep[1:] = np.any(line[1:] != line[:-1], axis=1)

    return line[keep]