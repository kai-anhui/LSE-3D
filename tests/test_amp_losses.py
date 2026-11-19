"""Exercise reliable gate edges under CUDA autocast, not just segmentation."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data.geometry import build_geometry
from models import LSE3D, ModelConfig
from models.losses import compute_loss
from utils.device import move


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gate_bce_autocast_matches_fp32_loss_and_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gate_logits = torch.tensor([[[0.0, 2.0, -1.0, 0.0]]], device=device, requires_grad=True)
    raw = gate_logits.sigmoid()  # ScopeBranch deliberately computes gates in fp32.
    logits = torch.zeros(1, 4, device=device, requires_grad=True)
    batch = move({
        "target": torch.tensor([[True, True, False, False]]),
        "valid": torch.ones(1, 4, dtype=torch.bool),
        "scene": {
            "area_w": torch.ones(4),
            "geometry": {"levels": [{
                "point2unit": torch.arange(4), "center": torch.zeros(4, 3),
                "scan_idx": torch.arange(4)[None],
                "edge_valid": torch.tensor([[False, True, True, True]]),
            }]},
        },
        "pairs": [],
    }, device)
    output = dict(prob=logits.sigmoid(), logits=logits, id_logit=logits, gate_raw=[raw])
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss, parts, counts = compute_loss(output, batch)
    reference_gate = F.binary_cross_entropy_with_logits(
        gate_logits[0, 0, 1:3], torch.tensor([1.0, 0.0], device=device))
    assert counts["gate"] == 1
    assert parts["gate"] == pytest.approx(reference_gate.item(), rel=1e-6)
    actual_grad = torch.autograd.grad(loss, gate_logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference_gate * .2, gate_logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_model_bf16_gate_supervision_optimizer_step():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA required")
    torch.manual_seed(11)
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    batch = move({
        "scene": {"xyz": torch.from_numpy(xyz), "area_w": torch.ones(4),
                  "geometry": build_geometry(xyz, voxel_size=.5)},
        "texts": ["the two points on the left"],
        "target": torch.tensor([[True, True, False, False]]),
        "valid": torch.ones(1, 4, dtype=torch.bool), "pairs": [],
    }, "cuda")
    model = LSE3D(ModelConfig(dim=16, hidden=8, state=4, gate_hidden=8,
                             scan_backend="parallel")).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.scopes[0].gates[0][-1].weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
            loss, parts, counts = compute_loss(output, batch)
        assert counts["gate"] == 1 and parts["gate"] > 0
        assert all(raw.dtype == torch.float32 for raw in output["gate_raw"])
        loss.backward()
        assert torch.isfinite(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    assert not torch.equal(before, model.scopes[0].gates[0][-1].weight)
