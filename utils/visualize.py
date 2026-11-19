"""Export predictions as colored PLY files, without a GUI dependency."""

from pathlib import Path
import json

import numpy as np
from scipy.spatial import cKDTree
from data.raw_cloud import open_coordinates, resolve_source_path


def save_prediction_ply(path, xyz, probability, threshold=0.5):
    xyz = np.asarray(xyz)
    probability = np.asarray(probability)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or probability.shape != (len(xyz),):
        raise ValueError("Expected xyz [M,3] and probability [M]")
    if not np.isfinite(xyz).all() or not np.isfinite(probability).all():
        raise ValueError("Coordinates and probabilities must be finite")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    matched = probability >= threshold
    rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    rgb[matched, 0] = 255
    rgb[~matched, 2] = 255
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        np.savetxt(stream, np.column_stack((xyz, rgb)), fmt="%.8f %.8f %.8f %d %d %d")


FULL_VERTEX_DTYPE = np.dtype([
    ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("probability", "<f4"),
])


def export_full_scene_ply(path, scene_dir, probability, threshold=0.5,
                          source_path=None, chunk_size=100000):
    """Color every original TXT/LAS point using its nearest sampled prediction.

    This is a visualization upsampling operation, not full-resolution network
    inference. Original order/float64 world coordinates are retained. Predictions
    at sampled source indices are preserved exactly, even with duplicate XYZ.
    """
    if not 0 <= threshold <= 1 or chunk_size <= 0:
        raise ValueError("Invalid threshold or chunk size")
    base, path = Path(scene_dir), Path(path)
    metadata = json.loads((base / "source.json").read_text(encoding="utf-8"))
    source = resolve_source_path(metadata["path"], base, source_path)
    if source.resolve() == path.resolve():
        raise ValueError("PLY output cannot overwrite the original point cloud")
    with open_coordinates(source) as xyz:
        return _export_full_scene(path, base, probability, threshold, source, metadata, xyz, chunk_size)


def _export_full_scene(path, base, probability, threshold, source, metadata, xyz, chunk_size):
    if len(xyz) != metadata["original_count"]:
        raise ValueError("Source point count does not match the imported scene")
    indices = np.load(base / "source_index.npy")
    local = np.load(base / "coord.npy")
    probability = np.asarray(probability, dtype=np.float32)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0
            or probability.shape != indices.shape or local.shape != (len(indices), 3)
            or np.any(indices < 0) or np.any(indices >= len(xyz))
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("Invalid sample indices or prediction dimensions")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    origin = np.asarray(metadata["origin"], dtype=np.float64)
    if origin.shape != (3,) or not np.allclose(xyz[indices] - origin, local, atol=1e-3, rtol=1e-6):
        raise ValueError("Source coordinates differ from the prepared scene")
    tree = cKDTree(xyz[indices] - origin)
    sort = np.argsort(indices)
    sorted_indices, sorted_prob = indices[sort], probability[sort]
    path.parent.mkdir(parents=True, exist_ok=True)
    count, max_distance = 0, 0.
    with path.open("wb") as stream:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            "comment Full source cloud; nearest sampled prediction; binary red/blue\n"
            f"element vertex {len(xyz)}\n"
            "property double x\nproperty double y\nproperty double z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property float probability\nend_header\n"
        )
        stream.write(header.encode("ascii"))
        for start in range(0, len(xyz), chunk_size):
            end = min(start + chunk_size, len(xyz))
            coords = xyz[start:end]
            if not np.isfinite(coords).all():
                raise ValueError("Nonfinite source coordinates")
            distance, nearest = tree.query(coords - origin, k=1)
            p = probability[nearest].copy()
            lo, hi = np.searchsorted(sorted_indices, [start, end])
            p[sorted_indices[lo:hi] - start] = sorted_prob[lo:hi]
            matched = p >= threshold
            vertices = np.zeros(end - start, dtype=FULL_VERTEX_DTYPE)
            for axis, field in enumerate(("x", "y", "z")):
                vertices[field] = coords[:, axis]
            vertices["red"][matched] = 255
            vertices["blue"][~matched] = 255
            vertices["probability"] = p
            vertices.tofile(stream)
            count += int(matched.sum())
            max_distance = max(max_distance, float(distance.max()))
    report = dict(source=str(source.resolve()), point_count=len(xyz),
                  sampled_point_count=len(indices), matched_points=count,
                  unmatched_points=len(xyz)-count, threshold=float(threshold),
                  mapping="nearest_sample_probability", coordinate_frame="original_world",
                  max_nearest_sample_distance_m=max_distance,
                  colors={"matched": [255, 0, 0], "unmatched": [0, 0, 255]})
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
