import json

import numpy as np
import pytest

from data.raw_cloud import resolve_source_path
from utils.visualize import export_full_scene_ply


def test_migrated_windows_path_exports_from_project_layout(tmp_path, monkeypatch):
    project = tmp_path / "project"
    scene = project / "outputs" / "prepared" / "scenes" / "site" / "site" / "block"
    scene.mkdir(parents=True)
    raw = project / "datasets_raw" / "point_data" / "迁移点云.txt"
    raw.parent.mkdir(parents=True)
    xyz = np.array([[400000., 4300000., 1000.], [400001., 4300000., 1000.]])
    np.savetxt(raw, xyz)
    np.save(scene / "coord.npy", (xyz - xyz[0]).astype(np.float32))
    np.save(scene / "source_index.npy", np.array([0, 1]))
    recorded = r"Z:\missing_previous_machine\datasets_raw\point_data\迁移点云.txt"
    (scene / "source.json").write_text(json.dumps({
        "path": recorded, "original_count": 2, "origin": xyz[0].tolist(),
    }), encoding="utf-8")
    # Works even when launched outside the project root.
    monkeypatch.chdir(tmp_path)
    report = export_full_scene_ply(tmp_path / "result.ply", scene, [.8, .1])
    assert report["source"] == str(raw.resolve())
    assert report["matched_points"] == 1 and report["unmatched_points"] == 1
    assert json.loads((scene / "source.json").read_text())["path"] == recorded


def test_override_is_authoritative_and_missing_override_fails(tmp_path):
    raw = tmp_path / "existing.txt"
    raw.write_text("0 0 0")
    assert resolve_source_path("missing.txt", tmp_path, raw) == raw.resolve()
    with pytest.raises(FileNotFoundError, match="--source"):
        resolve_source_path(str(raw), tmp_path, tmp_path / "wrong.txt")


def test_ambiguous_fallback_requires_explicit_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = tmp_path / "project" / "scene"
    scene.mkdir(parents=True)
    for root in (tmp_path, tmp_path / "project"):
        raw = root / "datasets_raw" / "point_data" / "ambiguous.txt"
        raw.parent.mkdir(parents=True)
        raw.write_text("0 0 0")
    with pytest.raises(ValueError, match="Multiple source"):
        resolve_source_path(r"Z:\old\ambiguous.txt", scene)


def test_missing_raw_cloud_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Copy the original file"):
        resolve_source_path("missing_0897ef.txt", tmp_path)
