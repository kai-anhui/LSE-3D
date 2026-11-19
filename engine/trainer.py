"""Training loop; command-line parsing lives in tools/train.py."""

from dataclasses import replace
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from data.dataset import SceneDataset
from data.transforms import augment
from data.build import build_dataloader
from models import ModelConfig, build_model
from models.losses import LossConfig, compute_loss
from utils.config import load_config
from utils.device import move
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.seed import seed_all
from utils.logger import log_metrics


def staged_loss(base, step, total):
    # A=0..20%, then EQ, change, inclusion ramps; C starts at 60%.
    def ramp(start):
        return min(1.0, max(0.0, (step / total - start) / 0.1))

    return replace(
        base,
        eq_w=base.eq_w * ramp(0.2),
        chg_w=base.chg_w * ramp(0.3),
        inc_w=base.inc_w * ramp(0.4),
        id_w=base.id_w * ramp(0.6),
        gate_w=base.gate_w * ramp(0.6),
    )


def train(args):
    config = load_config(args.config)
    cfg = ModelConfig(**config.get("model", {}))
    loss_cfg = LossConfig(**config.get("loss", {}))
    tc = config.get("train", {})
    seed = tc.get("seed", 0)
    seed_all(seed)
    torch.set_num_threads(tc.get("threads", 4))
    device = torch.device(args.device)
    steps = args.steps or tc.get("steps", 1000)
    dataset = SceneDataset(
        args.data, "train", config.get("voxel_size", 1.0), args.allow_unverified
    )
    if not dataset.selected:
        raise ValueError("Training split has no eligible queries")
    model = build_model(config).to(device)
    pre_ids = (
        {id(p) for p in model.text.encoder.parameters()} if cfg.text_model else set()
    )
    regular = [
        p for p in model.parameters() if p.requires_grad and id(p) not in pre_ids
    ]
    pretrained = [p for p in model.parameters() if p.requires_grad and id(p) in pre_ids]
    lr = tc.get("lr", 1e-4)
    opt = torch.optim.AdamW(
        [
            {"params": regular, "lr": lr, "base_lr": lr},
            {"params": pretrained, "lr": lr * 0.1, "base_lr": lr * 0.1},
        ],
        weight_decay=tc.get("weight_decay", 0.05),
    )
    rng = random.Random(seed)
    aug_rng = np.random.default_rng(seed)
    start = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        if checkpoint["config"] != config:
            raise ValueError("Resume configuration differs")
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        rng.setstate(checkpoint["sampler_rng"])
        if checkpoint.get("cuda_rng") and device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    config["train"] = dict(tc, seed=seed)  # provenance of default seed
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    accum = tc.get("accumulate", 1)
    loader = build_dataloader(config, dataset=dataset, split="train", rng=rng)
    batches = iter(loader)
    for step in range(start, steps):
        opt.zero_grad(set_to_none=True)
        summed = 0.0
        counts = {}
        parts = {}
        warm = max(1, int(steps * 0.05))
        factor = (
            (step + 1) / warm
            if step < warm
            else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        )
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * factor
        current = (
            staged_loss(loss_cfg, step, steps) if tc.get("staged", True) else loss_cfg
        )
        if cfg.states == "single" or cfg.gate_mode == "none":
            current = replace(current, gate_w=0.0)
        for micro in range(accum):
            batch = next(batches)
            if tc.get("augment", False):
                # Per-update seed makes resumed augmentation reproducible.
                aug_rng = np.random.default_rng(seed + step * accum + micro)
                batch = augment(batch, aug_rng, config.get("voxel_size", 1.0))
            batch = move(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=tc.get("bf16", False) and device.type == "cuda",
            ):
                output = model(batch)
                loss, parts, count = compute_loss(output, batch, current)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            (loss / accum).backward()
            summed += float(loss.detach()) / accum
            for k, v in count.items():
                counts[k] = counts.get(k, 0) + v
        gate_grad = sum(
            float(p.grad.float().norm())
            for name, p in model.named_parameters()
            if ".gates." in name and p.grad is not None
        )
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        opt.step()
        if step % tc.get("log_every", 10) == 0 or step == steps - 1:
            gate_values = []
            for raw, level in zip(
                output["gate_raw"], batch["scene"]["geometry"]["levels"]
            ):
                mask = level["edge_valid"][: raw.shape[1]]
                if mask.any():
                    gate_values.append(raw.detach()[:, mask].float().flatten())
            gate_stats = {}
            if gate_values:
                g = torch.cat(gate_values)
                gate_stats = {
                    "mean": float(g.mean()),
                    "min": float(g.min()),
                    "max": float(g.max()),
                    "count": len(g),
                }
            row = dict(
                step=step + 1,
                loss=summed,
                parts=parts,
                counts=counts,
                grad_norm=float(norm),
                gate_grad=gate_grad,
                gate=gate_stats,
                seconds=time.perf_counter() - begin,
                peak_cuda_gb=(torch.cuda.max_memory_allocated(device) / 1024**3
                              if device.type == "cuda" else None),
            )
            log_metrics(outdir / "train.jsonl", row)
        if (step + 1) % tc.get("save_every", 100) == 0 or step == steps - 1:
            save_checkpoint(
                outdir / "last.pt", model, opt, step + 1, config, sampler=rng
            )
    return model
