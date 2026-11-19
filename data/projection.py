"""Offline pinhole projection with observed-depth or z-buffer visibility."""

import numpy as np


def project_views(
    xyz,
    normals,
    intrinsics,
    world_to_camera,
    sizes,
    depths=None,
    top_k=4,
    tolerance=0.1,
    depth_decay=100.0,
):
    views = len(intrinsics)
    m = len(xyz)
    uv = np.zeros((views, m, 2), np.float32)
    weight = np.zeros((views, m), np.float32)
    for v, (K, T, (h, w)) in enumerate(zip(intrinsics, world_to_camera, sizes)):
        camera = xyz @ T[:3, :3].T + T[:3, 3]
        z = camera[:, 2]
        projected = camera @ K.T
        pixel = projected[:, :2] / np.maximum(z[:, None], 1e-8)
        ix = np.rint(pixel[:, 0]).astype(int)
        iy = np.rint(pixel[:, 1]).astype(int)
        valid = (z > 0) & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        candidates = np.flatnonzero(valid)
        if depths is None:
            depth = np.full((h, w), np.inf)
            np.minimum.at(depth, (iy[candidates], ix[candidates]), z[candidates])
        else:
            depth = depths[v]
        observed = depth[iy[candidates], ix[candidates]]
        valid[candidates] &= (
            np.isfinite(observed)
            & (observed > 0)
            & (np.abs(z[candidates] - observed) <= tolerance)
        )
        center = -T[:3, :3].T @ T[:3, 3]
        ray = center - xyz
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-8)
        confidence = np.abs((normals * ray).sum(1)) * np.exp(
            -np.maximum(z, 0) / depth_decay
        )
        weight[v] = np.where(valid, confidence, 0)
        uv[v] = pixel / np.array([max(w - 1, 1), max(h - 1, 1)]) * 2 - 1
    if views > top_k:
        discard = np.argsort(weight, axis=0)[: views - top_k]
        weight[discard, np.arange(m)[None]] = 0
    return uv, weight
