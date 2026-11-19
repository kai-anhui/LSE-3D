"""Replay mine-3dvg-v1 annotations using the existing annotation application's core.

Source files are read only. Every export retains source point indices and annotator
version. No relation or verified/negative certainty is inferred from text.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from data.dataset import write_jsonl


def import_mine(
    source,
    annotations,
    core_path,
    output,
    max_points=4096,
    seed=0,
    negative_policy="unknown",
):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory to preserve previous exports")
    output.mkdir(parents=True, exist_ok=True)
    # Reuse original parser and exact region semantics rather than guessing JSON geometry.
    sys.path.insert(0, str(Path(core_path).parent))
    spec = importlib.util.spec_from_file_location("mine_source_core", core_path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    cloud = np.loadtxt(source, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError("Expected XYZ [RGB] text cloud")
    original_count = len(cloud)
    rng = np.random.default_rng(seed)
    indices = (
        np.sort(rng.choice(len(cloud), min(max_points, len(cloud)), replace=False))
        if max_points
        else np.arange(len(cloud))
    )
    points = cloud[indices, :3]
    color = cloud[indices, 3:6] if cloud.shape[1] >= 6 else np.zeros_like(points)
    color = color / (65535 if color.max() > 255 else 255 if color.max() > 1 else 1)
    origin = (cloud[:, :3].min(0) + cloud[:, :3].max(0)) / 2
    del cloud
    site = scene = "changqing"
    block = "whole_scene_sample" if max_points else "whole_scene"
    base = output / "scenes" / site / scene / block
    base.mkdir(parents=True)
    np.save(base / "coord.npy", (points - origin).astype("float32"))
    np.save(base / "color.npy", color.astype("float32"))
    np.save(base / "area_w.npy", np.ones(len(points), np.float32))
    np.save(base / "source_index.npy", indices)
    (base / "source.json").write_text(
        json.dumps(
            dict(
                path=str(source),
                original_count=original_count,
                retained_count=len(points),
                origin=origin.tolist(),
                seed=seed,
                area_mode="point",
                negative_policy=negative_policy,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    queries = []
    report = {
        "versions": [],
        "duplicates": [],
        "empty_after_sampling": [],
        "missing_text": [],
        "negative_policy": negative_policy,
        "note": "Unreviewed source annotations; no strict pair supervision or independent-site test set.",
    }
    seen = {}
    masks = output / "masks" / scene / block
    masks.mkdir(parents=True)
    for file in sorted(Path(annotations).glob("*.json")):
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest in seen:
            report["duplicates"].append(dict(path=str(file), same_as=seen[digest]))
            continue
        seen[digest] = str(file)
        project = json.loads(file.read_text(encoding="utf-8-sig"))
        if project.get("schema") != "mine-3dvg-v1":
            raise ValueError("Unexpected annotation schema")
        if project["source"]["point_count"] != original_count:
            raise ValueError("Annotation source point count differs")
        report["versions"].append(
            dict(path=str(file), sha256=digest, instances=len(project["instances"]))
        )
        for inst in project["instances"]:
            texts = list(
                dict.fromkeys(
                    s.strip() for s in inst.get("descriptions", []) if s.strip()
                )
            )
            if not texts:
                report["missing_text"].append([digest[:12], inst["id"]])
                continue
            target = core.selection_mask(points, inst["operations"], project["origin"])
            if not target.any():
                report["empty_after_sampling"].append([digest[:12], inst["id"]])
                continue
            # Unknown negatives remain invalid unless explicitly opting into debug assumption.
            valid = (
                np.ones(len(target), bool)
                if negative_policy == "assume-complete"
                else target.copy()
            )
            for j, text in enumerate(texts):
                qid = f'cq_{digest[:12]}_{inst["id"]}_{j}'
                np.savez_compressed(
                    masks / (qid + ".npz"), target=target, valid=valid, ignore=~valid
                )
                queries.append(
                    dict(
                        query_id=qid,
                        site_id=site,
                        scene_id=scene,
                        block_id=block,
                        text=text,
                        lang="zh",
                        carrier_id=None,
                        scope_type="unknown",
                        verified=False,
                        source_version=digest,
                        source_instance=inst["id"],
                        class_name=inst["class_name"],
                        negative_policy=negative_policy,
                    )
                )
    ann = output / "annotations"
    ann.mkdir()
    write_jsonl(ann / "queries.jsonl", queries)
    write_jsonl(ann / "pairs.jsonl", [])
    (ann / "splits.json").write_text(
        json.dumps({"train": [site], "val": [], "test": []}), encoding="utf-8"
    )
    report["queries"] = len(queries)
    report["sample_points"] = len(points)
    (output / "import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--core", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-points", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--negative-policy", choices=["unknown", "assume-complete"], default="unknown"
    )
    a = p.parse_args()
    import_mine(
        a.source,
        a.annotations,
        a.core,
        a.output,
        a.max_points,
        a.seed,
        a.negative_policy,
    )
