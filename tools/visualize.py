"""Re-export an existing prediction as a full-source red/blue point cloud."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
from utils.visualize import export_full_scene_ply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    threshold = args.threshold
    sidecar = Path(args.prediction + ".json")
    if threshold is None and sidecar.exists():
        threshold = json.loads(sidecar.read_text(encoding="utf-8")).get("threshold")
    if threshold is None:
        threshold = .5
        print("No calibrated threshold supplied; using 0.5 for visualization.")
    with np.load(args.prediction) as data:
        probability = data["prob"]
        if "source_index" in data and not np.array_equal(
                data["source_index"], np.load(Path(args.scene) / "source_index.npy")):
            raise ValueError("Prediction indices do not match the prepared scene")
    report = export_full_scene_ply(args.output, args.scene, probability,
                                   threshold=threshold, source_path=args.source)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
