"""Scene-shared augmentation and geometric preprocessing."""

import numpy as np
import torch
from .geometry import build_geometry


def augment(batch, rng, voxel_size, drop_rate=0.05, image_drop=0.1):
    """Shared geometry changes and correspondence for all queries in one batch."""
    scene = batch["scene"]
    xyz = scene["xyz"].numpy().copy()
    color = scene["color"].numpy()
    angle = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    xyz = xyz @ rotation.T * rng.uniform(0.95, 1.05) + rng.normal(
        0, 0.005, xyz.shape
    ).astype("float32")
    keep = rng.random(len(xyz)) >= drop_rate
    if not keep.any():
        keep[0] = True
    xyz = xyz[keep]
    color = color[keep]
    new_scene = dict(
        scene,
        xyz=torch.from_numpy(xyz),
        color=torch.from_numpy(color),
        area_w=scene["area_w"][keep],
        geometry=build_geometry(xyz, color=color, voxel_size=voxel_size),
    )
    if "images" in scene:
        # Original correspondences move with their sampled source points.
        new_scene["uv"] = scene["uv"][:, keep]
        new_scene["view_weight"] = scene["view_weight"][:, keep].clone()
        if rng.random() < image_drop:
            new_scene["view_weight"].zero_()
    result = dict(batch, scene=new_scene)
    for key in ("target", "valid", "parent_target", "parent_valid"):
        if key in batch:
            result[key] = batch[key][:, keep]
    return result
