from __future__ import annotations

import pytest


def test_gap_and_order_penalty_use_paired_energy_layout() -> None:
    torch = pytest.importorskip("torch")
    from gnn_excited.losses import qm9gwbse_loss

    target = torch.tensor([[1.0, 0.1, 2.0, 0.2]])
    prediction = torch.tensor([[1.5, 0.1, 1.0, 0.2]], requires_grad=True)
    terms = qm9gwbse_loss(
        prediction,
        target,
        ("S1_eV", "log1p_S1_f", "S2_eV", "log1p_S2_f"),
        gap_weight=1.0,
        ordering_weight=1.0,
        return_components=True,
    )
    assert terms["energy_gap_mse"].item() == pytest.approx(2.25)
    assert terms["energy_order_penalty"].item() > 0
    terms["total"].backward()
    assert prediction.grad is not None


def test_visnet_one_pass_shape_backprop_and_frozen_encoder() -> None:
    pytest.importorskip("torch_geometric")
    torch = pytest.importorskip("torch")
    from gnn_excited.models.visnet import build_visnet, load_transfer_checkpoint

    kwargs = {
        "target_columns": ("S1_eV", "log1p_S1_f", "S2_eV", "log1p_S2_f"),
        "hidden_channels": 32,
        "num_layers": 1,
        "num_rbf": 8,
        "cutoff": 3.0,
        "max_num_neighbors": 8,
    }
    model = build_visnet(**kwargs)
    z = torch.tensor([6, 1, 1], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]])
    batch = torch.zeros(3, dtype=torch.long)
    output = model(z, pos, batch)
    assert output.shape == (1, 4)
    output.sum().backward()
    assert model.reduce_op == "mean"
    vector_projection = model.energy_head.output_network[0].vec1_proj.weight
    assert vector_projection.grad is not None
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)

    checkpoint = {"model_state_dict": model.state_dict()}
    restored = build_visnet(**kwargs)
    load_transfer_checkpoint(restored, checkpoint, mode="readout_only")
    encoder_key, encoder_weight = next((key, value) for key, value in checkpoint["model_state_dict"].items() if key.startswith("encoder."))
    assert torch.equal(restored.state_dict()[encoder_key], encoder_weight)
    assert all(not parameter.requires_grad for parameter in restored.encoder.parameters())
    assert any(parameter.requires_grad for parameter in restored.energy_head.parameters())
