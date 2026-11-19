"""Area-weighted supervision with applicability counts, never model inputs."""

from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class LossConfig:
    dice_w: float = 1.0
    eq_w: float = 1.0
    chg_w: float = 1.0
    chg_keep_w: float = 0.5
    inc_w: float = 0.5
    id_w: float = 0.3
    gate_w: float = 0.2


def mean(values, mask, weights):
    w = weights * mask
    return (values * w).sum() / w.sum().clamp_min(1e-8)


def verified(pair):
    return len(set(pair.get("verified_by", []))) >= 2 and pair.get("relation") in (
        "EQ",
        "SUB",
        "SUP",
        "EDIT",
    )


def compute_loss(output, batch, cfg=None):
    cfg = cfg or LossConfig()
    p, logits = output["prob"].float(), output["logits"].float()
    y, valid = batch["target"].float(), batch["valid"].bool()
    w = batch["scene"]["area_w"].float()
    zero = logits.sum() * 0
    terms = {k: [] for k in ("seg", "eq", "chg", "inc", "id", "gate")}
    counts = {"chg_changed": 0, "chg_retained": 0}
    for i in range(len(p)):
        if valid[i].any():
            bce = mean(
                F.binary_cross_entropy_with_logits(logits[i], y[i], reduction="none"),
                valid[i],
                w,
            )
            weight = w * valid[i]
            dice = zero
            if (y[i] * weight).sum() > 0:
                dice = 1 - (2 * (weight * p[i] * y[i]).sum() + 1e-6) / (
                    (weight * (p[i] + y[i])).sum() + 1e-6
                )
            terms["seg"].append(bce + cfg.dice_w * dice)
        if "parent_target" in batch and batch["parent_valid"][i].any():
            terms["id"].append(
                mean(
                    F.binary_cross_entropy_with_logits(
                        output["id_logit"][i].float(),
                        batch["parent_target"][i].float(),
                        reduction="none",
                    ),
                    batch["parent_valid"][i],
                    w,
                )
            )
    for pair in batch.get("pairs", []):
        if not verified(pair):
            continue
        a, b = pair["a"], pair["b"]
        v = valid[a] & valid[b]
        if not v.any():
            continue
        rel = pair["relation"]
        dp = p[b] - p[a]
        dy = y[b] - y[a]
        if rel == "EQ":
            terms["eq"].append(mean(dp.abs(), v, w))
        else:
            change = v & (dy != 0)
            keep = v & y[a].bool() & y[b].bool()
            if change.any():
                value = mean((dp - dy).abs(), change, w)
                counts["chg_changed"] += 1
                if keep.any():
                    value = value + cfg.chg_keep_w * mean(dp.abs(), keep, w)
                    counts["chg_retained"] += 1
                terms["chg"].append(value)
            if rel in ("SUB", "SUP"):
                violation = dp if rel == "SUB" else -dp
                terms["inc"].append(mean(violation.relu(), v, w))
        if pair.get("shared_carrier") and pair.get("shared_anchor"):
            terms["id"].append(
                mean(
                    (
                        output["id_logit"][a].sigmoid()
                        - output["id_logit"][b].sigmoid()
                    ).abs(),
                    v,
                    w,
                )
            )
    # Labels are reduced only here, never in SceneEncoder or ScopeBranch.
    if cfg.gate_w:
        for q in range(len(p)):
            edges_loss = []
            for raw, level in zip(
                output["gate_raw"], batch["scene"]["geometry"]["levels"]
            ):
                ids = level["point2unit"]
                n = len(level["center"])
                total = torch.bincount(ids, minlength=n)
                known = torch.zeros(n, device=p.device).index_add_(
                    0, ids, valid[q].float()
                )
                positives = torch.zeros(n, device=p.device).index_add_(
                    0, ids, (y[q] * valid[q]).float()
                )
                frac = positives / known.clamp_min(1)
                clear = (known == total) & ((frac <= 0.1) | (frac >= 0.9))
                label = frac >= 0.9
                for k in range(raw.shape[1]):
                    idx = level["scan_idx"][k]
                    current = label[idx]
                    previous = current.roll(1)
                    reliable = (
                        level["edge_valid"][k]
                        & clear[idx]
                        & clear[idx].roll(1)
                        & (current | previous)
                    )
                    reliable[0] = False
                    if reliable.any():
                        # ScopeBranch computes sigmoid gates in fp32. Keep their
                        # BCE in fp32 too: CUDA rejects probability BCE inside
                        # autocast even when its input is already float32.
                        # raw is a probability, NOT a logit.
                        with torch.autocast(device_type=raw.device.type, enabled=False):
                            edges_loss.append(
                                F.binary_cross_entropy(
                                    raw[q, k, reliable].float().clamp(1e-6, 1 - 1e-6),
                                    (current & previous)[reliable].float(),
                                    reduction="none",
                                )
                            )
            if edges_loss:
                terms["gate"].append(torch.cat(edges_loss).mean())
    values = {k: torch.stack(v).mean() if v else zero for k, v in terms.items()}
    counts.update({k: len(v) for k, v in terms.items()})
    total = values["seg"] + sum(
        getattr(cfg, k + "_w") * values[k] for k in ("eq", "chg", "inc", "id", "gate")
    )
    return total, {k: float(v.detach()) for k, v in values.items()}, counts
