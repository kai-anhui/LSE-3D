import torch
from torch import nn
from models.layers import mlp, SelectiveSSM


class ContextBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.project = mlp(3 * d, d, d)
        self.ssm = nn.ModuleList(
            [
                SelectiveSSM(d, d, cfg.hidden, cfg.state, cfg.scan_backend)
                for _ in range(cfg.directions)
            ]
        )
        self.fuse = nn.Linear(cfg.directions * cfg.hidden, d)

    def forward(self, z, q_id, q_anchor, level):
        n = z.shape[1]
        x = self.project(
            torch.cat(
                (
                    z,
                    q_id[:, None].expand(-1, n, -1),
                    q_anchor[:, None].expand(-1, n, -1),
                ),
                -1,
            )
        )
        outs = []
        for k, ssm in enumerate(self.ssm):
            seq = x[:, level["scan_idx"][k]]
            out = ssm(seq, seq, seq.new_ones(seq.shape[:2]))
            outs.append(out[:, level["inv_idx"][k]])
        return self.fuse(torch.cat(outs, -1))


class ScopeBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.gates = nn.ModuleList(
            [mlp(5 * d + 6, cfg.gate_hidden, 1) for _ in range(cfg.directions)]
        )
        for gate in self.gates:
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, 3.0)
        self.ssm = nn.ModuleList(
            [
                SelectiveSSM(3 * d, 2 * d, cfg.hidden, cfg.state, cfg.scan_backend)
                for _ in range(cfg.directions)
            ]
        )
        self.fuse = nn.Linear(cfg.directions * cfg.hidden, d)

    def forward(self, z, ctx, q_scope, level, intervention=None):
        outs, gates, raw = [], [], []
        for k, ssm in enumerate(self.ssm):
            idx = level["scan_idx"][k]
            zs, cs = z[:, idx], ctx[:, idx]
            zp = torch.cat((torch.zeros_like(zs[:, :1]), zs[:, :-1]), 1)
            cp = torch.cat((torch.zeros_like(cs[:, :1]), cs[:, :-1]), 1)
            q = q_scope[:, None].expand_as(zs)
            gate_q = q if self.cfg.gate_use_language else torch.zeros_like(q)
            edge = level["edge_feat"][k][None].expand(z.shape[0], -1, -1)
            features = torch.cat((zs, zp, edge, cs, cp, gate_q), -1)
            # Gate and recurrence remain float32 even under encoder autocast.
            with torch.autocast(device_type=z.device.type, enabled=False):
                r = self.gates[k](features.float()).squeeze(-1).sigmoid()
            if intervention == "ones":
                r = torch.ones_like(r)
            elif intervention == "shuffle":
                r = r[:, torch.randperm(r.shape[1], device=r.device)]
            rho = r * level["edge_valid"][k]
            if self.cfg.gate_mode != "recurrent":
                rho = torch.ones_like(rho)
            out = ssm(torch.cat((zs, cs, q), -1), torch.cat((zs, q), -1), rho)
            outs.append(out[:, level["inv_idx"][k]])
            gates.append(rho)
            raw.append(r)
        return (
            self.fuse(torch.cat(outs, -1)),
            torch.stack(gates, 1),
            torch.stack(raw, 1),
        )
