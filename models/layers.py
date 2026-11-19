"""Exact fp32 gated selective recurrence; no log-gate approximation."""

import torch
from torch import nn
from torch.nn import functional as F


def affine_scan(a, b, valid=None, parallel=False):
    # [B,N,H,S]; padding is the identity transition, with zero emitted state.
    if valid is not None:
        a = torch.where(valid[..., None, None], a, torch.ones_like(a))
        b = torch.where(valid[..., None, None], b, torch.zeros_like(b))
    if parallel:
        offset = 1
        while offset < a.shape[1]:
            b = torch.cat(
                (b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]), 1
            )
            a = torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), 1)
            offset *= 2
        out = b
    else:
        h = torch.zeros_like(b[:, 0])
        states = []
        for t in range(a.shape[1]):
            h = a[:, t] * h + b[:, t]
            states.append(h)
        out = torch.stack(states, 1)
    return out if valid is None else out * valid[..., None, None]


def selective_scan(u, dt, bw, cr, A, rho, valid=None, backend="reference", chunk=64):
    u, dt, bw, cr, A, rho = [x.float() for x in (u, dt, bw, cr, A, rho)]
    if backend not in ("reference", "parallel", "chunked"):
        raise ValueError(backend)
    if backend == "chunked":
        # Keep carry across chunks, exact zeros supported. Avoid full-sequence states.
        carry = u.new_zeros(u.shape[0], u.shape[2], A.shape[1])
        outputs = []
        for start in range(0, u.shape[1], chunk):
            end = min(start + chunk, u.shape[1])
            sl = slice(start, end)
            a = rho[:, sl, None, None] * torch.exp(dt[:, sl, :, None] * A)
            b = dt[:, sl, :, None] * bw[:, sl, None, :] * u[:, sl, :, None]
            for t in range(end - start):
                h = a[:, t] * carry + b[:, t]
                if valid is not None:
                    h = torch.where(valid[:, start + t, None, None], h, carry)
                carry = h
                o = (h * cr[:, start + t, None, :]).sum(-1)
                if valid is not None:
                    o = o * valid[:, start + t, None]
                outputs.append(o)
        return torch.stack(outputs, 1)
    a = rho[..., None, None] * torch.exp(dt[..., None] * A)
    b = dt[..., None] * bw[:, :, None, :] * u[..., None]
    h = affine_scan(a, b, valid, backend == "parallel")
    return (h * cr[:, :, None, :]).sum(-1)


class SelectiveSSM(nn.Module):
    def __init__(self, input_dim, param_dim, hidden, state, backend="chunked"):
        super().__init__()
        self.u = nn.Linear(input_dim, hidden)
        self.params = nn.Linear(param_dim, hidden + 2 * state)
        self.theta = nn.Parameter(
            torch.arange(1, state + 1).float().log().repeat(hidden, 1)
        )
        self.hidden, self.state, self.backend = hidden, state, backend

    def forward(self, inputs, conditioning, rho, valid=None):
        u = self.u(inputs)
        dt, bw, cr = self.params(conditioning).split(
            [self.hidden, self.state, self.state], -1
        )
        return selective_scan(
            u,
            F.softplus(dt.float()),
            bw,
            cr,
            -self.theta.float().exp(),
            rho,
            valid,
            self.backend,
        )


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class RolePooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.ModuleList([nn.Linear(dim, 1) for _ in range(3)])
        self.project = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])

    def forward(self, tokens, valid):
        roles, attention = [], []
        for score, project in zip(self.score, self.project):
            a = score(tokens).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(-1)
            attention.append(a)
            roles.append(project((tokens * a[..., None]).sum(1)))
        return roles, torch.stack(attention, 1)


def pool_points(feat, ids, n):
    sums = feat.new_zeros(n, feat.shape[-1]).index_add(0, ids, feat)
    mean = sums / torch.bincount(ids, minlength=n).clamp_min(1)[:, None]
    maximum = feat.new_full((n, feat.shape[-1]), -torch.inf).scatter_reduce(
        0, ids[:, None].expand_as(feat), feat, reduce="amax", include_self=True
    )
    return torch.cat((mean, maximum), -1)
