"""Validation calibration and evaluation using frozen thresholds."""

import json
from pathlib import Path
import numpy as np
import torch
from data.dataset import SceneDataset
from data.build import build_dataloader
from models import build_model
from models.losses import verified
from utils.device import move
from utils.checkpoint import load_checkpoint
from utils.metrics import single_metrics, pair_metrics, aggregate


@torch.no_grad()
def collect(model, dataset, device, query_chunk=1):
    model.eval()
    records = []
    # Encoding shared within each block; query chunk bounds activation memory.
    for batch in build_dataloader({}, dataset=dataset, split="val"):
        scene = move(batch["scene"], device)
        encoded = model.encode_scene(scene)
        outputs = []
        for start in range(0, len(batch["texts"]), query_chunk):
            out = model(
                {"scene": scene, "texts": batch["texts"][start : start + query_chunk]},
                encoded_scene=encoded,
            )
            outputs.append(out["prob"].cpu().numpy())
        records.append((batch, np.concatenate(outputs)))
    return records


def score(records, threshold):
    rows = []
    for batch, p in records:
        y = batch["target"].numpy()
        v = batch["valid"].numpy()
        w = batch["scene"]["area_w"].numpy()
        for i, meta in enumerate(batch["metadata"]):
            metrics = single_metrics(p[i], y[i], v[i], w, threshold)
            if meta.get("scope_type") in ("full", "local"):
                metrics.update(
                    {
                        k + "_" + meta["scope_type"]: value
                        for k, value in list(metrics.items())
                    }
                )
            rows.append(dict(meta, metrics=metrics))
        for pair in batch["pairs"]:
            if not verified(pair):
                continue
            a, b = pair["a"], pair["b"]
            meta = batch["metadata"][a]
            metrics = pair_metrics(
                p[a], p[b], y[a], y[b], v[a] & v[b], w, pair["relation"], threshold
            )
            metrics.update(
                {f'{k}_{pair["relation"]}': value for k, value in list(metrics.items())}
            )
            rows.append(dict(meta, metrics=metrics))
    return aggregate(rows)


def evaluate(args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(args.checkpoint, args.device)
    model = build_model(ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    if args.command == "calibrate" and args.split != "val":
        raise ValueError("Threshold calibration is validation-only")
    dataset = SceneDataset(
        args.data,
        args.split,
        ckpt["config"].get("voxel_size", 1.0),
        args.allow_unverified,
    )
    if not dataset.selected:
        raise ValueError("No eligible queries in requested split")
    records = collect(model, dataset, args.device)
    if args.command == "calibrate":
        candidates = [
            (t, score(records, t).get("mIoU_A", {}).get("mean", float("-inf")))
            for t in np.linspace(0.1, 0.9, 17)
        ]
        threshold, value = max(
            candidates, key=lambda item: (item[1], -abs(item[0] - 0.5))
        )
        if not np.isfinite(value):
            raise ValueError("No nonempty valid validation targets")
        ckpt["threshold"] = float(threshold)
        ckpt["calibration"] = {
            "split": "val",
            "score": value,
            "sites": sorted({q["site_id"] for q in dataset.selected}),
        }
        torch.save(ckpt, args.output)
        print(json.dumps(ckpt["calibration"] | {"threshold": threshold}))
        return
    if args.command == "predict":
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        for batch, p in records:
            for q, prob in zip(batch["metadata"], p):
                np.save(path / (q["query_id"] + ".npy"), prob)
        return
    if ckpt.get("threshold") is None:
        raise ValueError("Calibrate on validation data before evaluation")
    result = score(records, ckpt["threshold"])
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
