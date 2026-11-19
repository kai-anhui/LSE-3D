"""Generate deterministic synthetic sites for integration tests only."""

import argparse
import json
from pathlib import Path
import numpy as np
from data.dataset import write_jsonl


def make_demo(root):
    root = Path(root)
    queries = []
    pairs = []
    for split in ("train", "val", "test"):
        site = "synthetic_" + split
        scene = site
        block = "full"
        base = root / "scenes" / site / scene / block
        base.mkdir(parents=True, exist_ok=True)
        x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 4))
        xyz = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(
            "float32"
        )
        np.save(base / "coord.npy", xyz)
        np.save(base / "area_w.npy", np.ones(len(xyz), np.float32))
        masks = [xyz[:, 0] > -0.5, xyz[:, 0] > 0.35, xyz[:, 0] > -0.5]
        texts = [
            "the broad right region",
            "the narrow right region",
            "the broad right region",
        ]
        ids = []
        for i, (mask, text) in enumerate(zip(masks, texts)):
            qid = site + f"_q{i}"
            ids.append(qid)
            queries.append(
                dict(
                    query_id=qid,
                    site_id=site,
                    scene_id=scene,
                    block_id=block,
                    text=text,
                    carrier_id=site + "_carrier",
                    scope_type="full" if i != 1 else "local",
                    verified=True,
                )
            )
            folder = root / "masks" / scene / block
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                folder / (qid + ".npz"),
                target=mask,
                valid=np.ones(len(mask), bool),
                ignore=np.zeros(len(mask), bool),
            )
        for a, b, rel in ((0, 1, "SUB"), (0, 2, "EQ")):
            pairs.append(
                dict(
                    pair_id=site + rel,
                    q_a=ids[a],
                    q_b=ids[b],
                    relation=rel,
                    shared_carrier=True,
                    shared_anchor=True,
                    verified_by=["synthetic_oracle_1", "synthetic_oracle_2"],
                )
            )
    ann = root / "annotations"
    ann.mkdir(exist_ok=True)
    write_jsonl(ann / "queries.jsonl", queries)
    write_jsonl(ann / "pairs.jsonl", pairs)
    (ann / "splits.json").write_text(
        json.dumps({s: ["synthetic_" + s] for s in ("train", "val", "test")}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    make_demo(p.parse_args().output)
