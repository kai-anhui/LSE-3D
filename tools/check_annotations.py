import argparse
import json
from collections import defaultdict
from data.dataset import SceneDataset


def check(root):
    report = {
        "queries": 0,
        "pairs": 0,
        "unverified_queries": 0,
        "unverified_pairs": 0,
        "duplicate_text_groups": [],
    }
    descriptions = defaultdict(list)
    for split in ("train", "val", "test"):
        ds = SceneDataset(root, split, allow_unverified=True)
        for q in ds.selected:
            y, v = ds.mask(q)
            if len(y) != len(ds.scene(q)["xyz"]):
                raise ValueError(f'Mask length: {q["query_id"]}')
            report["queries"] += 1
            report["unverified_queries"] += not q.get("verified", False)
            descriptions[(q["scene_id"], q["block_id"], q["text"])].append(
                q["query_id"]
            )
        for pair in ds.pairs:
            report["pairs"] += 1
            if len(set(pair.get("verified_by", []))) < 2:
                report["unverified_pairs"] += 1
                continue
            ya, va = ds.mask(ds.by_id[pair["q_a"]])
            yb, vb = ds.mask(ds.by_id[pair["q_b"]])
            v = va & vb
            plus, minus = (~ya & yb & v).any(), (ya & ~yb & v).any()
            relation = pair["relation"]
            ok = {
                "EQ": not plus and not minus,
                "SUB": not plus and bool(minus),
                "SUP": bool(plus) and not minus,
                "EDIT": bool(plus) and bool(minus),
                "OTHER": True,
                "UNKNOWN": True,
            }[relation]
            if not ok:
                raise ValueError(f'Relation/mask inconsistency: {pair["pair_id"]}')
    report["duplicate_text_groups"] = [
        dict(scene=k[0], block=k[1], text=k[2], queries=v)
        for k, v in descriptions.items()
        if len(v) > 1
    ]
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    args = p.parse_args()
    print(json.dumps(check(args.data), ensure_ascii=False, indent=2))
