"""Query-independent surface construction and reproducible directed scans."""

import numpy as np
import torch
from scipy.spatial import cKDTree


def describe(xyz, k=16):
    """PCA normals and linearity/planarity/scattering, O(M k) memory."""
    _, ix = cKDTree(xyz).query(xyz, k=min(k, len(xyz)))
    ix = np.asarray(ix).reshape(len(xyz), -1)
    local = xyz[ix] - xyz[ix].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", local, local) / ix.shape[1]
    val, vec = np.linalg.eigh(cov)
    normal = vec[:, :, 0]
    normal *= np.where(normal[:, 2:3] < 0, -1, 1)
    a, b, c = val[:, 2], val[:, 1], val[:, 0]
    shape = np.stack((a - b, b - c, c), -1) / np.maximum(a[:, None], 1e-10)
    return normal.astype("float32"), shape.astype("float32")


def hilbert_code(coords, bits=10):
    """3D Hilbert transpose transform, integer grid -> scalar ordering."""
    x = coords.copy().astype(np.int64)
    q = 1 << (bits - 1)
    while q > 1:
        p = q - 1
        for i in range(3):
            mask = (x[:, i] & q) != 0
            t = (x[:, 0] ^ x[:, i]) & p
            x[:, 0] ^= np.where(mask, p, t)
            x[:, i] ^= np.where(mask, 0, t)
        q >>= 1
    x[:, 1] ^= x[:, 0]
    x[:, 2] ^= x[:, 1]
    t = np.zeros(len(x), dtype=np.int64)
    q = 1 << (bits - 1)
    while q > 1:
        t ^= np.where((x[:, 2] & q) != 0, q - 1, 0)
        q >>= 1
    x ^= t[:, None]
    code = np.zeros(len(x), dtype=np.int64)
    for b in range(bits - 1, -1, -1):
        for axis in range(3):
            code = (code << 1) | ((x[:, axis] >> b) & 1)
    return code


def scans(center, normal, radius, k=8):
    grid = np.floor(
        (center - center.min(0)) / np.maximum(np.ptp(center, axis=0), 1e-9) * 1023
    ).astype(np.int64)
    morton = np.zeros(len(center), dtype=np.int64)
    for b in range(10):
        for a in range(3):
            morton |= ((grid[:, a] >> b) & 1) << (3 * b + a)
    orders = [
        np.argsort(hilbert_code(grid), kind="stable"),
        np.argsort(morton, kind="stable"),
    ]
    idx = np.stack([orders[0], orders[0][::-1], orders[1], orders[1][::-1]]).copy()
    _, nn = cKDTree(center).query(center, k=min(k + 1, len(center)))
    nn = np.asarray(nn).reshape(len(center), -1)
    edge = np.zeros((4, len(center), 6), dtype=np.float32)
    valid = np.zeros((4, len(center)), dtype=bool)
    for j, order in enumerate(idx):
        a, b = order[1:], order[:-1]
        delta = center[a] - center[b]
        dist = np.linalg.norm(delta, axis=-1)
        angle = np.arccos(np.clip((normal[a] * normal[b]).sum(-1), -1, 1))
        edge[j, 1:] = np.column_stack((delta, dist, angle, delta[:, 2]))
        adjacent = (nn[a] == b[:, None]).any(1) | (nn[b] == a[:, None]).any(1)
        valid[j, 1:] = adjacent & (dist < radius)
    return dict(
        scan_idx=idx, inv_idx=np.argsort(idx, axis=1), edge_feat=edge, edge_valid=valid
    )


def build_geometry(
    xyz, normal=None, color=None, voxel_size=1.0, normal_cos=0.9, color_distance=0.25
):
    """Fine voxel region growing; coarse adjacency clustering, all label-free.

    Region sizes are bounded (8 voxels / 8 fine units) to prevent a smooth
    connected surface collapsing into one giant unit. This is an engineering default.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("A scene must contain finite points")
    estimated, shape = describe(xyz)
    normal = estimated if normal is None else np.asarray(normal, dtype=np.float32)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    color = np.zeros_like(xyz) if color is None else np.asarray(color)
    grid, pvox = np.unique(
        np.floor((xyz - xyz.min(0)) / voxel_size).astype(np.int64),
        axis=0,
        return_inverse=True,
    )

    def means(values, ids):
        n = ids.max() + 1
        out = np.zeros((n, values.shape[1]))
        np.add.at(out, ids, values)
        return out / np.bincount(ids)[:, None]

    vc, vn, vr = [means(v, pvox) for v in (xyz, normal, color)]
    lookup = {tuple(g): i for i, g in enumerate(grid)}
    adjacency = [[] for _ in grid]
    for i, g in enumerate(grid):
        for axis in range(3):
            for sign in (-1, 1):
                h = g.copy()
                h[axis] += sign
                j = lookup.get(tuple(h))
                if (
                    j is not None
                    and vn[i] @ vn[j] >= normal_cos
                    and np.linalg.norm(vr[i] - vr[j]) <= color_distance
                ):
                    adjacency[i].append(j)

    def cluster(adj, limit):
        ids = np.full(len(adj), -1, dtype=np.int64)
        group = 0
        for start in range(len(adj)):
            if ids[start] >= 0:
                continue
            ids[start] = group
            queue = [start]
            for node in queue:
                for other in sorted(adj[node]):
                    if ids[other] < 0 and len(queue) < limit:
                        ids[other] = group
                        queue.append(other)
            group += 1
        return ids

    fine = cluster(adjacency, 8)[pvox]
    centers = means(xyz, fine)
    dist, ix = cKDTree(centers).query(centers, k=min(9, len(centers)))
    dist, ix = dist.reshape(len(centers), -1), ix.reshape(len(centers), -1)
    coarse_adj = [
        set(row[d < voxel_size * 6].tolist()) - {i}
        for i, (row, d) in enumerate(zip(ix, dist))
    ]
    for i, neighbors in enumerate(list(map(set, coarse_adj))):
        for j in neighbors:
            coarse_adj[j].add(i)
    coarse = cluster(coarse_adj, 8)[fine]
    levels = []
    for level, ids in enumerate((fine, coarse)):
        center, nu, sh = [means(v, ids).astype("float32") for v in (xyz, normal, shape)]
        nu /= np.maximum(np.linalg.norm(nu, axis=1, keepdims=True), 1e-8)
        distance, nearest = cKDTree(center).query(xyz, k=min(4, len(center)))
        nearest = nearest.reshape(len(xyz), -1)
        distance = distance.reshape(len(xyz), -1)
        weight = 1 / np.maximum(distance, 1e-6)
        weight /= weight.sum(-1, keepdims=True)
        entry = dict(
            point2unit=ids,
            center=center,
            normal=nu,
            shape=sh,
            interp_idx=nearest,
            interp_w=weight.astype("float32"),
        )
        entry.update(scans(center, nu, voxel_size * (4 if level == 0 else 10)))
        levels.append({k: torch.from_numpy(v) for k, v in entry.items()})
    return dict(
        normal=torch.from_numpy(normal), shape=torch.from_numpy(shape), levels=levels
    )
