import numpy as np
from .pi2.pi2py2 import *


def get_skeleton_data(img: np.ndarray) -> list[np.ndarray]:
    """
    Vectorize 1-pixel skeleton image using pi2.

    Input:
        img: 2D numpy array.
             Skeleton pixels are non-zero.

    Output:
        [
            np.array([[x1, y1], [x2, y2], ...], dtype=np.int32),
            ...
        ]
    """
    img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={img.shape}")

    pi = Pi2()

    skel_np = (img > 0).astype(np.uint8) * 255

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
        True,  # store all edge points
        1,     # thread count
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

    points_xyz = _normalize_points_xyz(points.to_numpy())
    return _parse_lines_to_polylines(lines.to_numpy(), points_xyz)


def _normalize_points_xyz(points_np: np.ndarray) -> np.ndarray:
    """
    pi2 getpointsandlines returns points as x,y,z coordinates.

    Output:
        shape (N, 3)
    """
    arr = np.asarray(points_np)
    arr = np.squeeze(arr)

    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"Cannot interpret points array shape={arr.shape}")
        return arr.reshape(-1, 3).astype(np.float32)

    if arr.ndim == 2:
        # pi2 docs say points are 3 x N.
        # Prefer 3 x N interpretation.
        if arr.shape[0] == 3:
            return arr.T.astype(np.float32)

        # fallback for wrappers that return N x 3
        if arr.shape[1] == 3:
            return arr.astype(np.float32)

    raise ValueError(f"Cannot interpret points array shape={arr.shape}")


def _parse_lines_to_polylines(
    lines_np: np.ndarray,
    points_xyz: np.ndarray,
) -> list[np.ndarray]:
    """
    Parse compressed pi2 lines.

    Format:
        [line_count,
         point_count_0, idx_0_0, idx_0_1, ...,
         point_count_1, idx_1_0, idx_1_1, ...,
         ...]

    Output:
        [
            np.array([[x1, y1], [x2, y2], ...], dtype=np.int32),
            ...
        ]
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

        if point_indices.size == 0:
            continue

        if point_indices.min() < 0 or point_indices.max() >= len(points_xyz):
            raise ValueError(
                f"Bad point index in line {line_idx}: "
                f"min={point_indices.min()}, max={point_indices.max()}, "
                f"points_count={len(points_xyz)}"
            )

        coords_xyz = points_xyz[point_indices]

        # pi2 через from_numpy() в вашем случае ведет себя как row,col,z.
        # Для формата [[x, y], ...] нужно row,col -> col,row.
        xy = coords_xyz[:, [1, 0]]

        # pi2 can return centroid points with fractional coordinates.
        xy = np.rint(xy).astype(np.int32)

        xy = _remove_consecutive_duplicate_points(xy)

        if len(xy) > 0:
            polylines.append(xy)

    return polylines


def _remove_consecutive_duplicate_points(line: np.ndarray) -> np.ndarray:
    if len(line) <= 1:
        return line

    keep = np.ones(len(line), dtype=bool)
    keep[1:] = np.any(line[1:] != line[:-1], axis=1)
    return line[keep]