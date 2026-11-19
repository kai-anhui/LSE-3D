import torch
from torch import nn
from .config import ModelConfig
from .layers import mlp, pool_points, RolePooling
from .backbone import TextEncoder, SceneEncoder, ContextBranch, ScopeBranch
from .heads import DualHeadDecoder


class LSE3D(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or ModelConfig()
        c = self.cfg
        if c.states not in ("dual", "single", "scope_only"):
            raise ValueError(c.states)
        if c.gate_mode not in ("none", "recurrent"):
            raise ValueError(
                "Output gate baseline requires a separately specified graph operator"
            )
        if c.directions not in (1, 2, 4):
            raise ValueError("directions must be 1, 2, or 4")
        self.scene = SceneEncoder(c.dim)
        self.text = TextEncoder(c)
        self.roles = RolePooling(c.dim)
        self.units = nn.ModuleList([mlp(2 * c.dim + 9, c.dim, c.dim) for _ in range(2)])
        self.contexts = nn.ModuleList([ContextBranch(c) for _ in range(2)])
        self.scopes = nn.ModuleList([ScopeBranch(c) for _ in range(2)])
        self.single_query = nn.Linear(3 * c.dim, c.dim)
        self.decoder = DualHeadDecoder(c.dim)

    def encode_scene(self, scene):
        feat, avail = self.scene(scene)
        zs = []
        for proj, level in zip(self.units, scene["geometry"]["levels"]):
            pooled = pool_points(feat, level["point2unit"], len(level["center"]))
            zs.append(
                proj(
                    torch.cat(
                        (pooled, level["center"], level["normal"], level["shape"]), -1
                    )
                )
            )
        return feat, avail, zs

    def forward(self, batch, encoded_scene=None, intervention=None):
        # Only scene and text are read here; annotations never enter model inputs.
        feat, avail, zs = (
            self.encode_scene(batch["scene"])
            if encoded_scene is None
            else encoded_scene
        )
        tokens, valid = self.text(batch["texts"])
        roles, alpha = self.roles(tokens, valid)
        qi, qa, qr = roles
        if intervention == "swap_scope":
            qr = qr.roll(1, 0)
        contexts, scopes, gates, raws = [], [], [], []
        levels = batch["scene"]["geometry"]["levels"]
        for z, context, scope, level in zip(zs, self.contexts, self.scopes, levels):
            z = z[None].expand(len(qi), -1, -1)
            if self.cfg.states == "single":
                joint = self.single_query(torch.cat(roles, -1))
                ctx = context(z, joint, torch.zeros_like(joint), level)
                out = ctx
                gate = raw = z.new_zeros(len(qi), self.cfg.directions, z.shape[1])
            else:
                ctx = (
                    context(z, qi, qa, level)
                    if self.cfg.states == "dual"
                    else torch.zeros_like(z)
                )
                out, gate, raw = scope(z, ctx, qr, level, intervention)
            contexts.append(ctx)
            scopes.append(out)
            gates.append(gate)
            raws.append(raw)
        identity, delta = self.decoder(contexts, scopes, feat, levels)
        logits = identity + delta
        return dict(
            logits=logits,
            prob=logits.sigmoid(),
            id_logit=identity,
            scope_delta=delta,
            gate=gates,
            gate_raw=raws,
            aux=dict(role_attention=alpha, img_avail=avail, scope_feat=scopes),
        )
