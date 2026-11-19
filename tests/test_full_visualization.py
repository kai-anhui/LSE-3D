import json

import numpy as np
import pytest

from utils.visualize import FULL_VERTEX_DTYPE, export_full_scene_ply, save_prediction_ply


def read_binary_ply(path):
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        return header, np.fromfile(stream, dtype=FULL_VERTEX_DTYPE)


@pytest.fixture
def scene(tmp_path):
    xyz = np.array([[443900., 4364000., 1200.], [443901., 4364000., 1200.],
                    [443910., 4364000., 1200.], [443909., 4364000., 1200.],
                    [443910., 4364000., 1200.]])
    source = tmp_path / "cloud.txt"
    np.savetxt(source, xyz)
    indices = np.array([4, 0, 2])  # unsorted, duplicate coordinates, distinct indices
    origin = xyz[0]
    np.save(tmp_path / "coord.npy", (xyz[indices]-origin).astype(np.float32))
    np.save(tmp_path / "source_index.npy", indices)
    (tmp_path / "source.json").write_text(json.dumps({"path": str(source),
        "origin": origin.tolist(), "original_count": len(xyz)}))
    return tmp_path, xyz


def test_full_export_preserves_coordinates_and_sample_indices(scene):
    base, xyz = scene
    out = base / "full.ply"
    report = export_full_scene_ply(out, base, [.8, .2, .3], threshold=.5, chunk_size=2)
    header, vertices = read_binary_ply(out)
    assert "element vertex 5" in header
    np.testing.assert_array_equal(np.column_stack([vertices[k] for k in ('x','y','z')]), xyz)
    np.testing.assert_allclose(vertices['probability'][[4,0,2]], [.8,.2,.3])
    assert vertices['probability'][1] == pytest.approx(.2)
    matched = vertices['probability'] >= .5
    np.testing.assert_array_equal(vertices['red'], matched*255)
    np.testing.assert_array_equal(vertices['blue'], (~matched)*255)
    assert not vertices['green'].any()
    assert report['matched_points'] + report['unmatched_points'] == len(xyz)


def test_full_export_rejects_wrong_source(scene):
    base, _ = scene
    changed = base / 'wrong.txt'
    np.savetxt(changed, np.zeros((5,3)))
    with pytest.raises(ValueError, match='coordinates differ'):
        export_full_scene_ply(base/'bad.ply', base, [.8,.2,.3], source_path=changed)
    assert not (base/'bad.ply').exists()
    with pytest.raises(ValueError, match='overwrite'):
        export_full_scene_ply(base/'cloud.txt', base, [.8,.2,.3])


def test_binary_colors_at_threshold(tmp_path):
    out = tmp_path/'sample.ply'
    save_prediction_ply(out, np.zeros((2,3)), [.49,.5], threshold=.5)
    rows = out.read_text().split('end_header\n')[1].splitlines()
    assert rows[0].endswith('0 0 255') and rows[1].endswith('255 0 0')
