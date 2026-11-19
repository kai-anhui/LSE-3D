"""Selection replay adapted from the local mine annotation application core."""
import numpy as np


def polygon_contains(xy, polygon):
    """Ray crossing, including the boundary; supports concave polygons."""
    xy = np.asarray(xy, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    if len(poly) < 3:
        return np.zeros(len(xy), dtype=bool)
    x, y = xy.T
    inside = np.zeros(len(xy), dtype=bool)
    edge = inside.copy()
    previous = poly[-1]
    for current in poly:
        x1, y1 = previous
        x2, y2 = current
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length > 1e-14:
            cross = (x - x1) * dy - (y - y1) * dx
            edge |= (np.abs(cross) <= 1e-9 * length) & (x >= min(x1, x2) - 1e-9) & (x <= max(x1, x2) + 1e-9) & (y >= min(y1, y2) - 1e-9) & (y <= max(y1, y2) + 1e-9)
        if abs(dy) > 1e-14:
            inside ^= ((y1 > y) != (y2 > y)) & (x < dx * (y - y1) / dy + x1)
        previous = current
    return inside | edge


def region_mask(xyz, region, origin):
    matrix = np.asarray(region["matrix"], dtype=np.float64).reshape(4, 4)
    local = np.asarray(xyz, dtype=np.float64) - np.asarray(origin)
    clip = local @ matrix[:, :3].T + matrix[:, 3]
    valid = clip[:, 3] > 1e-12
    ndc = np.zeros((len(xyz), 3), dtype=np.float64)
    np.divide(clip[:, :3], clip[:, 3, None], out=ndc, where=valid[:, None])
    valid &= np.all((ndc >= -1 - 1e-9) & (ndc <= 1 + 1e-9), axis=1)
    if region.get("z_range") is not None:
        lo, hi = region["z_range"]
        valid &= (xyz[:, 2] >= lo) & (xyz[:, 2] <= hi)
    hits = np.flatnonzero(valid)
    valid[hits] = polygon_contains(ndc[hits, :2], region["polygon"])
    return valid


def selection_mask(xyz, operations, origin):
    selected = np.zeros(len(xyz), dtype=bool)
    for op in operations:
        mask = region_mask(xyz, op, origin)
        action = op["operation"]
        if action == "replace":
            selected = mask
        elif action == "add":
            selected |= mask
        elif action == "subtract":
            selected &= ~mask
        elif action == "intersect":
            selected &= mask
        else:
            raise ValueError(f"未知选区运算：{action}")
    return selected
