import random

import numpy as np
import pytest
import torch

from data import SceneDataset, build_dataloader
from models import build_model
from tools.make_demo import make_demo
from utils.config import load_config
from utils.visualize import save_prediction_ply


def test_yaml_inheritance_and_factory(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "model:\n  dim: 16\n  hidden: 8\n  state: 3\ntrain:\n  seed: 2\n"
    )
    (tmp_path / "exp.yaml").write_text("_base_: base.yaml\nmodel:\n  directions: 2\n")
    config = load_config(tmp_path / "exp.yaml")
    model = build_model(config)
    assert model.cfg.dim == 16 and model.cfg.directions == 2
    assert config["train"]["seed"] == 2
    (tmp_path / "cycle.yaml").write_text("_base_: cycle.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(tmp_path / "cycle.yaml")


def test_scene_loader_preserves_sampling_and_rng(tmp_path):
    make_demo(tmp_path)
    ds = SceneDataset(tmp_path, voxel_size=0.4)
    assert len(ds) == 3 and len(ds[0]["texts"]) == 1
    expected_rng = random.Random(123)
    loader_rng = random.Random(123)
    loader = build_dataloader({}, dataset=ds, rng=loader_rng)
    state = torch.get_rng_state().clone()
    batches = iter(loader)
    for _ in range(4):
        expected, actual = ds.sample(expected_rng), next(batches)
        assert expected["texts"] == actual["texts"]
        torch.testing.assert_close(expected["target"], actual["target"])
        assert expected["pairs"] == actual["pairs"]
    assert torch.equal(state, torch.get_rng_state())
    assert loader_rng.getstate() == expected_rng.getstate()
    evaluation = list(build_dataloader({}, dataset=ds, split="val"))
    assert len(evaluation) == 1 and len(evaluation[0]["texts"]) == 3


def test_visualization_export(tmp_path):
    path = tmp_path / "result.ply"
    save_prediction_ply(path, np.zeros((2, 3)), np.array([0.0, 1.0]))
    text = path.read_text()
    assert "element vertex 2" in text and "0 0 255" in text and "255 0 0" in text
