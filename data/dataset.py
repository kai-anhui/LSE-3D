"""Disk schema, strict site splits and scene-shared query sampling."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .geometry import build_geometry


def read_jsonl(path):
    return [
        json.loads(s)
        for s in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if s.strip()
    ]


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def block_key(q):
    return q["site_id"], q["scene_id"], q["block_id"]


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", voxel_size=1.0, allow_unverified=False):
        self.root = Path(root)
        self.voxel_size = voxel_size
        ann = self.root / "annotations"
        self.queries = read_jsonl(ann / "queries.jsonl")
        self.by_id = {q["query_id"]: q for q in self.queries}
        if len(self.by_id) != len(self.queries):
            raise ValueError("Duplicate query IDs")
        splits = json.loads((ann / "splits.json").read_text(encoding="utf-8"))
        if split not in splits:
            raise ValueError(f"No split {split}")
        seen = set()
        for sites in splits.values():
            if seen.intersection(sites):
                raise ValueError("Physical site leakage between splits")
            seen.update(sites)
        if any(q["site_id"] not in seen for q in self.queries):
            raise ValueError("Unassigned site")
        self.selected = [
            q
            for q in self.queries
            if q["site_id"] in splits[split]
            and (allow_unverified or q.get("verified", False))
        ]
        selected = {q["query_id"] for q in self.selected}
        all_pairs = read_jsonl(ann / "pairs.jsonl")
        self.pairs = []
        for p in all_pairs:
            a, b = self.by_id[p["q_a"]], self.by_id[p["q_b"]]
            if block_key(a) != block_key(b):
                raise ValueError("Pair crosses scene blocks")
            if p["relation"] not in ("EQ", "SUB", "SUP", "EDIT", "OTHER", "UNKNOWN"):
                raise ValueError("Unknown relation")
            if p["q_a"] in selected and p["q_b"] in selected:
                self.pairs.append(p)
        self.cache = {}

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, index):
        """A single query sample; balanced pairs use build_dataloader."""
        return self.batch([self.selected[index]])

    def scene(self, q):
        key = block_key(q)
        if key not in self.cache:
            base = self.root / "scenes" / key[0] / key[1] / key[2]
            xyz = np.load(base / "coord.npy").astype("float32")
            w = np.load(base / "area_w.npy").astype("float32")
            if w.shape != (len(xyz),) or not np.isfinite(w).all() or (w <= 0).any():
                raise ValueError("Invalid area weights")
            color = (
                np.load(base / "color.npy").astype("float32")
                if (base / "color.npy").exists()
                else np.zeros_like(xyz)
            )
            normal = (
                np.load(base / "normal.npy").astype("float32")
                if (base / "normal.npy").exists()
                else None
            )
            digest = hashlib.sha256(
                xyz.tobytes()
                + color.tobytes()
                + (b"" if normal is None else normal.tobytes())
                + str(self.voxel_size).encode()
                + b"geometry-v1"
            ).hexdigest()[:16]
            cache = base / "cache"
            cache.mkdir(exist_ok=True)
            path = cache / f"geometry-{digest}.pt"
            if path.exists():
                geometry = torch.load(path, weights_only=True)
            else:
                geometry = build_geometry(xyz, normal, color, self.voxel_size)
                torch.save(geometry, path)
            scene = {
                "xyz": torch.from_numpy(xyz),
                "color": torch.from_numpy(color),
                "area_w": torch.from_numpy(w),
                "geometry": geometry,
            }
            # Optional preprocessed calibrated image tensors, no annotation content.
            if (base / "views.pt").exists():
                views = torch.load(base / "views.pt", weights_only=True)
                scene.update({k: views[k] for k in ("images", "uv", "view_weight")})
            self.cache[key] = scene
        return self.cache[key]

    def mask(self, q, parent=False):
        path = q.get("parent_mask") if parent else None
        if parent and not path:
            return None
        path = (
            self.root / path
            if path
            else self.root
            / "masks"
            / q["scene_id"]
            / q["block_id"]
            / (q["query_id"] + ".npz")
        )
        with np.load(path) as d:
            y = d["target"].astype(bool)
            v = d["valid"].astype(bool)
            if "ignore" in d:
                v &= ~d["ignore"].astype(bool)
        if y.shape != v.shape or y.ndim != 1:
            raise ValueError("Invalid mask dimensions")
        return torch.from_numpy(y), torch.from_numpy(v)

    def batch(self, queries, pairs=None):
        if len({block_key(q) for q in queries}) != 1:
            raise ValueError("One physical block per batch")
        scene = self.scene(queries[0])
        masks = [self.mask(q) for q in queries]
        if any(len(y) != len(scene["xyz"]) for y, v in masks):
            raise ValueError("Point/mask mismatch")
        parents = [self.mask(q, True) for q in queries]
        result = dict(
            scene=scene,
            texts=[q["text"] for q in queries],
            target=torch.stack([m[0] for m in masks]),
            valid=torch.stack([m[1] for m in masks]),
            metadata=queries,
            pairs=[],
        )
        if any(x is not None for x in parents):
            default = torch.zeros(len(scene["xyz"]), dtype=torch.bool)
            result["parent_target"] = torch.stack(
                [p[0] if p else default for p in parents]
            )
            result["parent_valid"] = torch.stack(
                [p[1] if p else default for p in parents]
            )
        indices = {q["query_id"]: i for i, q in enumerate(queries)}
        for p in pairs or []:
            result["pairs"].append(dict(p, a=indices[p["q_a"]], b=indices[p["q_b"]]))
        return result

    def sample(self, rng):
        if not self.selected:
            raise ValueError(
                "No eligible queries: review data or use explicit --allow-unverified for debugging"
            )
        eligible = [p for p in self.pairs if len(set(p.get("verified_by", []))) >= 2]
        if eligible and rng.random() < 0.5:
            groups = {p["relation"] for p in eligible}
            relation = rng.choice(sorted(groups))
            p = rng.choice([p for p in eligible if p["relation"] == relation])
            return self.batch([self.by_id[p["q_a"]], self.by_id[p["q_b"]]], [p])
        return self.batch([rng.choice(self.selected)])

    def evaluation_batches(self):
        groups = {}
        for q in self.selected:
            groups.setdefault(block_key(q), []).append(q)
        for queries in groups.values():
            ids = {q["query_id"] for q in queries}
            yield self.batch(
                queries, [p for p in self.pairs if p["q_a"] in ids and p["q_b"] in ids]
            )
