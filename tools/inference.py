"""Label-free inference for a new query and a prepared scene directory."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from data.geometry import build_geometry
from utils.device import move
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.visualize import save_prediction_ply, export_full_scene_ply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--intervention", choices=["ones", "shuffle"])
    p.add_argument("--visualize", help="Optional colored PLY output path")
    p.add_argument("--visualize-scope", choices=["auto", "full", "sampled"], default="auto",
                   help="auto exports the full cloud when source.json is available")
    p.add_argument("--source", help="Override the original XYZ TXT path for full export")
    p.add_argument("--threshold", type=float, help="Match threshold; checkpoint value or 0.5 by default")
    args = p.parse_args()
    base = Path(args.scene)
    ckpt = load_checkpoint(args.checkpoint, args.device)
    threshold = args.threshold if args.threshold is not None else ckpt.get("threshold")
    threshold_source = "command_line" if args.threshold is not None else "validation_checkpoint"
    if threshold is None:
        threshold, threshold_source = 0.5, "uncalibrated_default"
    if not 0 <= threshold <= 1:
        p.error("--threshold must be between 0 and 1")
    model = build_model(ckpt["config"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    xyz = np.load(base / "coord.npy").astype("float32")
    color = (
        np.load(base / "color.npy").astype("float32")
        if (base / "color.npy").exists()
        else np.zeros_like(xyz)
    )
    geometry = build_geometry(
        xyz, color=color, voxel_size=ckpt["config"].get("voxel_size", 1.0)
    )
    scene = dict(
        xyz=torch.from_numpy(xyz), color=torch.from_numpy(color), geometry=geometry
    )
    if (base / "views.pt").exists():
        views = torch.load(base / "views.pt", weights_only=True)
        scene.update({k: views[k] for k in ("images", "uv", "view_weight")})
    with torch.no_grad():
        result = model(
            {"scene": move(scene, args.device), "texts": [args.text]},
            intervention=args.intervention,
        )
    arrays = {
        "prob": result["prob"][0].cpu().numpy(),
        "id_logit": result["id_logit"][0].cpu().numpy(),
        "scope_delta": result["scope_delta"][0].cpu().numpy(),
    }
    arrays["mask"] = arrays["prob"] >= threshold
    if (base / "source_index.npy").exists():
        arrays["source_index"] = np.load(base / "source_index.npy")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    visualization = None
    if args.visualize:
        full = args.visualize_scope == "full" or (
            args.visualize_scope == "auto" and (base / "source.json").exists())
        if full:
            visualization = export_full_scene_ply(args.visualize, base, arrays["prob"], threshold, args.source)
        else:
            save_prediction_ply(args.visualize, xyz, arrays["prob"], threshold)
            visualization = {"point_count": len(xyz), "mapping": "sampled_only"}
        print(json.dumps(visualization, ensure_ascii=False))
    Path(args.output + ".json").write_text(
        json.dumps(
            {
                "query": args.text,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "visualization": visualization,
                "point_count": len(xyz),
                "intervention": args.intervention,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
