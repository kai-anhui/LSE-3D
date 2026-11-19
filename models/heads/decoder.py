import torch
from torch import nn
from models.layers import mlp


class DualHeadDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ctx = nn.Linear(2 * dim, dim)
        self.scope = nn.Linear(2 * dim, dim)
        self.identity = mlp(2 * dim, dim, 1)
        self.residual = mlp(3 * dim, dim, 1)

    def forward(self, contexts, scopes, pt_feat, levels):
        def interpolate(xs):
            return torch.cat(
                [
                    (x[:, lev["interp_idx"]] * lev["interp_w"][None, :, :, None]).sum(2)
                    for x, lev in zip(xs, levels)
                ],
                -1,
            )

        c, r = self.ctx(interpolate(contexts)), self.scope(interpolate(scopes))
        p = pt_feat[None].expand(c.shape[0], -1, -1)
        identity = self.identity(torch.cat((c, p), -1)).squeeze(-1)
        delta = self.residual(torch.cat((r, c, p), -1)).squeeze(-1)
        return identity, delta
