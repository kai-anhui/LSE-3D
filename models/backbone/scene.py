import torch
from torch import nn
from models.layers import mlp


class SceneEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.geometry = mlp(13, dim, dim)  # xyz, normal, PCA shape, local height, RGB
        self.image = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, padding=1),
        )
        self.fuse = mlp(2 * dim + 1, dim, dim)

    def forward(self, scene):
        xyz = scene["xyz"]
        geo = scene["geometry"]
        x = xyz - xyz.mean(0)
        height = xyz[:, 2:3] - xyz[:, 2:3].min()
        g = self.geometry(
            torch.cat(
                (
                    x,
                    geo["normal"],
                    geo["shape"],
                    height,
                    scene.get("color", torch.zeros_like(x)),
                ),
                -1,
            )
        )
        img = torch.zeros_like(g)
        avail = torch.zeros(len(xyz), device=xyz.device, dtype=torch.bool)
        if "images" in scene:
            # uv [V,M,2] normalized grid_sample coordinates; confidence already visibility-masked.
            maps = self.image(scene["images"])
            sampled = (
                nn.functional.grid_sample(
                    maps, scene["uv"][:, :, None], align_corners=True
                )
                .squeeze(-1)
                .permute(0, 2, 1)
            )
            weights = scene["view_weight"].float()
            if (weights < 0).any():
                raise ValueError("Negative visibility confidence")
            sampled = torch.where(
                weights[..., None] > 0, sampled, torch.zeros_like(sampled)
            )
            denom = weights.sum(0)
            avail = denom > 0
            img = (sampled * weights[..., None]).sum(0) / denom.clamp_min(1e-8)[:, None]
            img = torch.where(avail[:, None], img, torch.zeros_like(img))
        return self.fuse(torch.cat((g, img, avail[:, None].to(g.dtype)), -1)), avail
