import inspect
import numpy as np
import pytest
import torch
from models.layers import affine_scan, selective_scan
from models import LSE3D, ModelConfig
from models.backbone import ContextBranch
from data.dataset import SceneDataset
from data.geometry import build_geometry, hilbert_code
from models.losses import compute_loss, LossConfig
from utils.metrics import pair_metrics, boundary_fscore
from data.projection import project_views
from tools.make_demo import make_demo


@pytest.fixture
def batch(tmp_path):
    make_demo(tmp_path)
    ds = SceneDataset(tmp_path, voxel_size=0.4)
    return ds.batch(ds.selected, ds.pairs)


def test_scan_backends_and_gradients():
    torch.manual_seed(0)
    a = torch.rand(2, 9, 3, 2, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    x = affine_scan(a, b)
    y = affine_scan(a, b, parallel=True)
    torch.testing.assert_close(x, y)
    gx = torch.autograd.grad(x.square().sum(), (a, b), retain_graph=True)
    gy = torch.autograd.grad(y.square().sum(), (a, b))
    for i, j in zip(gx, gy):
        torch.testing.assert_close(i, j)


def test_zero_gate_and_padding():
    a = torch.ones(1, 6, 1, 1) * 0.8
    a[:, 3] = 0
    b = torch.ones_like(a)
    changed = b.clone()
    changed[:, :3] = 100
    torch.testing.assert_close(affine_scan(a, b)[:, 3:], affine_scan(a, changed)[:, 3:])
    valid = torch.tensor([[True, True, False, True, True, False]])
    result = affine_scan(a, b, valid)
    torch.testing.assert_close(
        result[:, valid[0]], affine_scan(a[:, valid[0]], b[:, valid[0]])
    )
    assert not result[:, ~valid[0]].any()


def test_selective_discretization():
    torch.manual_seed(1)
    u = torch.randn(2, 7, 3)
    dt = torch.rand_like(u)
    bw = torch.randn(2, 7, 2)
    cr = torch.randn_like(bw)
    A = -torch.rand(3, 2)
    rho = torch.rand(2, 7)
    ref = selective_scan(u, dt, bw, cr, A, rho)
    for mode in ("parallel", "chunked"):
        torch.testing.assert_close(
            ref, selective_scan(u, dt, bw, cr, A, rho, backend=mode, chunk=3)
        )
    expected = []
    h = torch.zeros(2, 3, 2)
    for t in range(7):
        h = (
            torch.exp(dt[:, t, :, None] * A) * h
            + dt[:, t, :, None] * bw[:, t, None, :] * u[:, t, :, None]
        )
        expected.append((h * cr[:, t, None, :]).sum(-1))
    torch.testing.assert_close(
        torch.stack(expected, 1), selective_scan(u, dt, bw, cr, A, torch.ones_like(rho))
    )


def test_optional_mamba_reference():
    module = pytest.importorskip("mamba_ssm.ops.selective_scan_interface")
    torch.manual_seed(2)
    u = torch.randn(1, 8, 4)
    dt = torch.rand_like(u)
    b = torch.randn(1, 8, 3)
    c = torch.randn_like(b)
    a = -torch.rand(4, 3)
    expected = module.selective_scan_ref(
        u.transpose(1, 2), dt.transpose(1, 2), a, b.transpose(1, 2), c.transpose(1, 2)
    )
    actual = selective_scan(u, dt, b, c, a, torch.ones(1, 8))
    torch.testing.assert_close(actual, expected.transpose(1, 2), rtol=1e-4, atol=1e-5)


def test_hilbert_and_geometry(batch):
    grid = np.stack(
        np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij"), -1
    ).reshape(-1, 3)
    code = hilbert_code(grid, bits=2)
    assert len(np.unique(code)) == 64
    assert (np.abs(np.diff(grid[np.argsort(code)], axis=0)).sum(1) == 1).all()
    for level in batch["scene"]["geometry"]["levels"]:
        idx, inv = level["scan_idx"], level["inv_idx"]
        n = idx.shape[1]
        assert torch.equal(torch.gather(idx, 1, inv), torch.arange(n).expand(4, -1))
        assert not level["edge_valid"][:, 0].any()
        torch.testing.assert_close(
            level["edge_feat"][0, 1:, :3], -level["edge_feat"][1, 1:, :3].flip(0)
        )
    fresh = build_geometry(batch["scene"]["xyz"].numpy(), voxel_size=0.4)
    for a, b in zip(fresh["levels"], batch["scene"]["geometry"]["levels"]):
        assert torch.equal(a["scan_idx"], b["scan_idx"])


def test_model_backward_isolation_and_label_leakage(batch):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = LSE3D(
        ModelConfig(dim=16, hidden=8, state=3, gate_hidden=8, scan_backend="parallel")
    )
    out = model(batch)
    assert out["prob"].shape == batch["target"].shape
    assert "q_scope" not in inspect.signature(ContextBranch.forward).parameters
    grads = torch.autograd.grad(
        out["id_logit"].sum(),
        out["aux"]["scope_feat"],
        allow_unused=True,
        retain_graph=True,
    )
    assert all(g is None for g in grads)
    assert not out["aux"]["img_avail"].any()
    for gate, raw, lev in zip(
        out["gate"], out["gate_raw"], batch["scene"]["geometry"]["levels"]
    ):
        torch.testing.assert_close(
            raw, torch.full_like(raw, torch.sigmoid(torch.tensor(3.0)).item())
        )
        assert not gate[:, ~lev["edge_valid"]].any()
    changed = dict(batch, target=~batch["target"], valid=~batch["valid"], pairs=[])
    torch.testing.assert_close(out["prob"], model(changed)["prob"])
    single = model(dict(batch, texts=batch["texts"][:1]))["prob"]
    torch.testing.assert_close(single, out["prob"][:1], atol=1e-6, rtol=1e-5)
    loss, _, counts = compute_loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss) and counts["chg"] == 1
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if ".gates." in n
    )


def test_loss_empty_and_direction(batch):
    p = torch.tensor([[0.8, 0.2], [0.2, 0.1]], requires_grad=True)
    small = dict(
        batch,
        target=torch.tensor([[True, False], [False, False]]),
        valid=torch.ones(2, 2, dtype=torch.bool),
        scene={"area_w": torch.ones(2)},
        pairs=[dict(a=0, b=1, relation="SUB", verified_by=["a", "b"])],
    )
    out = {
        "prob": p,
        "logits": torch.logit(p),
        "id_logit": torch.logit(p),
        "gate_raw": [],
    }
    cfg = LossConfig(gate_w=0)
    loss, terms, count = compute_loss(out, small, cfg)
    assert terms["chg"] == pytest.approx(0.4) and terms["inc"] == 0
    reversed_batch = dict(
        small,
        target=small["target"].flip(0),
        pairs=[dict(a=0, b=1, relation="SUP", verified_by=["a", "b"])],
    )
    reversed_out = {
        k: v.flip(0) if isinstance(v, torch.Tensor) else v for k, v in out.items()
    }
    loss2, terms2, _ = compute_loss(reversed_out, reversed_batch, cfg)
    torch.testing.assert_close(loss, loss2)
    empty = dict(small, valid=torch.zeros(2, 2, dtype=torch.bool))
    zero, _, counts = compute_loss(out, empty, cfg)
    assert zero == 0 and counts["seg"] == 0
    zero.backward()
    assert torch.isfinite(p.grad).all()


def test_metrics_hand_examples():
    ya = np.array([1, 1, 0], bool)
    yb = np.zeros(3, bool)
    pa = np.array([1, 1, 0.0])
    pb = np.array([0, 0, 1.0])
    v = np.ones(3, bool)
    w = np.ones(3)
    metrics = pair_metrics(pa, pb, ya, yb, v, w, "SUB")
    assert metrics["DirIoU"] == pytest.approx(2 / 3)
    assert metrics["FalseChange"] == pytest.approx(1 / 3)
    assert pair_metrics(pb * 0, pb * 0, yb, yb, v, w, "EQ")["EQCons"] == 1
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert boundary_fscore(yb, yb, v, xyz, np.array([[0, 1], [1, 2]]), 0.1) is None
    assert boundary_fscore(yb, ya, v, xyz, np.array([[0, 1], [1, 2]]), 0.1) == 0


def test_split_leak(tmp_path):
    import json

    make_demo(tmp_path)
    (tmp_path / "annotations" / "splits.json").write_text(
        json.dumps(
            {"train": ["synthetic_train"], "val": ["synthetic_train"], "test": []}
        )
    )
    with pytest.raises(ValueError, match="leakage"):
        SceneDataset(tmp_path)


def test_projection_visibility():
    xyz = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [5.0, 0.0, 1.0]])
    uv, w = project_views(
        xyz, np.tile([0, 0, 1.0], (3, 1)), [np.eye(3)], [np.eye(4)], [(3, 3)]
    )
    assert w[0, 0] > 0 and w[0, 1] == 0 and w[0, 2] == 0


def test_image_missing_and_present(batch):
    model = LSE3D(ModelConfig(dim=16, hidden=8, state=3, gate_hidden=8))
    scene = dict(batch["scene"])
    m = len(scene["xyz"])
    scene.update(
        images=torch.randn(1, 3, 5, 5),
        uv=torch.zeros(1, m, 2),
        view_weight=torch.zeros(1, m),
    )
    a, avail = model.scene(scene)
    b, _ = model.scene(batch["scene"])
    torch.testing.assert_close(a, b)
    assert not avail.any()
    scene["view_weight"][0, :3] = 1
    c, avail = model.scene(scene)
    assert avail.sum() == 3
    c.sum().backward()
    assert model.scene.image[0].weight.grad.abs().sum() > 0


def test_gate_aux_reliable_edges(batch):
    ids = torch.arange(4)
    level = {
        "point2unit": ids,
        "center": torch.zeros(4, 3),
        "scan_idx": ids[None],
        "edge_valid": torch.tensor([[False, True, True, True]]),
    }
    raw = torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], requires_grad=True)
    small = dict(
        batch,
        target=torch.tensor([[True, True, False, False]]),
        valid=torch.ones(1, 4, dtype=torch.bool),
        scene={"area_w": torch.ones(4), "geometry": {"levels": [level]}},
        pairs=[],
    )
    logits = torch.zeros(1, 4, requires_grad=True)
    out = dict(prob=logits.sigmoid(), logits=logits, id_logit=logits, gate_raw=[raw])
    loss, parts, counts = compute_loss(out, small)
    assert parts["gate"] == pytest.approx(np.log(2)) and counts["gate"] == 1
    loss.backward()
    assert raw.grad[0, 0, 1] < 0 and raw.grad[0, 0, 2] > 0 and raw.grad[0, 0, 3] == 0


@pytest.mark.parametrize(
    "states,mode", [("single", "none"), ("scope_only", "recurrent"), ("dual", "none")]
)
def test_ablation_forward(batch, states, mode):
    model = LSE3D(
        ModelConfig(
            dim=16,
            hidden=8,
            state=3,
            gate_hidden=8,
            states=states,
            gate_mode=mode,
            scan_backend="parallel",
        )
    )
    out = model(batch)
    loss, _, _ = compute_loss(out, batch, LossConfig(gate_w=0))
    loss.backward()
    assert torch.isfinite(loss)


def test_deterministic_updates(batch):
    def run():
        torch.manual_seed(7)
        model = LSE3D(
            ModelConfig(
                dim=16, hidden=8, state=3, gate_hidden=8, scan_backend="parallel"
            )
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        history = []
        for _ in range(3):
            optimizer.zero_grad()
            loss, _, _ = compute_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        return history

    assert run() == run()
