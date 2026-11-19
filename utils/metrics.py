"""Fixed-reference weighted metrics. None means not applicable."""

from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def iou(a, b, valid, weights):
    union = weights[valid & (a | b)].sum()
    return float(weights[valid & a & b].sum() / union) if union else 1.0


def single_metrics(prob, target, valid, weights, threshold=0.5):
    pred = prob >= threshold
    if not valid.any():
        return {}
    if not (target & valid).any():
        return {"empty_correct": float(not (pred & valid).any())}
    j = iou(pred, target, valid, weights)
    jp = iou(pred, target, valid, np.ones_like(weights))
    return {
        "mIoU_A": j,
        "mIoU_P": jp,
        "Acc25": float(jp >= 0.25),
        "Acc50": float(jp >= 0.5),
    }


def pair_metrics(pa, pb, ya, yb, valid, w, relation, threshold=0.5):
    if not valid.any():
        return {}
    a, b = pa >= threshold, pb >= threshold
    out = {}
    if relation == "EQ":
        out["EQCons"] = iou(a, b, valid, w)
        if (ya & valid).any() and (yb & valid).any():
            out["EQJoint50"] = float(
                iou(a, ya, valid, w) >= 0.5 and iou(b, yb, valid, w) >= 0.5
            )
    plus, minus = ~ya & yb & valid, ya & ~yb & valid
    pp, pm = ~a & b & valid, a & ~b & valid
    if relation in ("SUB", "SUP", "EDIT") and (plus | minus).any():
        denom = w[plus | pp].sum() + w[minus | pm].sum()
        out["DirIoU"] = float((w[plus & pp].sum() + w[minus & pm].sum()) / denom)
        changed, predicted = plus | minus, pp | pm
        out["FalseChange"] = float(
            w[predicted & ~changed].sum() / w[changed | predicted].sum()
        )
    keep = ya & yb & valid
    if keep.any():
        out["Drift_fg"] = float(
            (w[keep] * np.abs(pa[keep] - pb[keep])).sum() / w[keep].sum()
        )
    return out


def boundary_fscore(pred, target, valid, xyz, edges, delta, excluded=None):
    """Geodesics on supplied reference-surface edges (not Euclidean shortcuts).

    Caller supplies a mesh/surface adjacency and excludes crop/uncertain margins.
    """
    eligible = valid.copy()
    if excluded is not None:
        eligible &= ~excluded
    a, b = np.asarray(edges).T
    use = eligible[a] & eligible[b]
    a, b = a[use], b[use]

    def boundary(mask):
        cross = mask[a] != mask[b]
        return np.unique(np.concatenate((a[cross], b[cross])))

    truth, prediction = boundary(target), boundary(pred)
    if not len(truth):
        return None
    if not len(prediction):
        return 0.0
    lengths = np.linalg.norm(xyz[a] - xyz[b], axis=-1)
    graph = coo_matrix(
        (np.r_[lengths, lengths], (np.r_[a, b], np.r_[b, a])),
        shape=(len(xyz), len(xyz)),
    ).tocsr()
    dist_t = dijkstra(graph, directed=False, indices=truth, min_only=True)
    dist_p = dijkstra(graph, directed=False, indices=prediction, min_only=True)
    precision, recall = (dist_t[prediction] <= delta).mean(), (
        dist_p[truth] <= delta
    ).mean()
    return (
        float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )


def aggregate(rows):
    keys = set().union(*(r["metrics"] for r in rows)) if rows else set()
    result = {}
    for key in sorted(keys):
        groups = defaultdict(lambda: defaultdict(list))
        values = []
        for row in rows:
            value = row["metrics"].get(key)
            if value is not None:
                values.append(value)
                groups[row["site_id"]][
                    row.get("carrier_id") or row.get("query_id", "unknown")
                ].append(value)
        sites = {
            s: float(np.mean([np.mean(v) for v in parents.values()]))
            for s, parents in groups.items()
        }
        result[key] = {
            "mean": float(np.mean(values)),
            "count": len(values),
            "site_macro": float(np.mean(list(sites.values()))),
            "sites": sites,
        }
    return result


def bootstrap_sites(site_values, seed=0, repeats=1000):
    if len(site_values) < 2:
        return None
    x = np.array(list(site_values.values()))
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, (repeats, len(x)), replace=True).mean(1)
    return np.quantile(samples, [0.025, 0.975]).tolist()
