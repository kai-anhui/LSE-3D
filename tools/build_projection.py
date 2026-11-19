"""Input NPZ: xyz, normal, intrinsics[V,3,3], world_to_camera[V,4,4],
images[V,H,W,3] uint8/float, optional depths[V,H,W]. Saves views.pt for SceneDataset.
"""

import argparse
import numpy as np
import torch
from data.projection import project_views


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-k", type=int, default=4)
    args = p.parse_args()
    with np.load(args.input) as data:
        images = data["images"]
        sizes = [images.shape[1:3]] * len(images)
        uv, w = project_views(
            data["xyz"],
            data["normal"],
            data["intrinsics"],
            data["world_to_camera"],
            sizes,
            data["depths"] if "depths" in data else None,
            args.top_k,
        )
        images = images.astype("float32") / (255 if images.dtype == np.uint8 else 1)
        torch.save(
            dict(
                images=torch.from_numpy(images).permute(0, 3, 1, 2),
                uv=torch.from_numpy(uv),
                view_weight=torch.from_numpy(w),
            ),
            args.output,
        )
