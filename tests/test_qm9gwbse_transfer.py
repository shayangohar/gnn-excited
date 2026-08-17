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


def test_spectroscopy_decoder_physics_phase_loss_and_encoder_transfer() -> None:
    pytest.importorskip("torch_geometric")
    torch = pytest.importorskip("torch")
    from gnn_excited.losses import qm9gwbse_loss
    from gnn_excited.models.visnet import OSCILLATOR_STRENGTH_FACTOR, build_visnet, load_transfer_checkpoint

    columns = tuple(
        column
        for state in range(1, 6)
        for column in (f"S{state}_eV", f"log1p_S{state}_f")
    )
    kwargs = {
        "target_columns": columns,
        "hidden_channels": 32,
        "num_layers": 1,
        "num_rbf": 8,
        "cutoff": 3.0,
        "max_num_neighbors": 8,
    }
    pretrained = build_visnet(**kwargs)
    model = build_visnet(**kwargs, spectroscopy_decoder=True)
    load_transfer_checkpoint(
        model,
        {"model_state_dict": pretrained.state_dict()},
        mode="encoder_finetune",
    )
    encoder_key = next(key for key in pretrained.state_dict() if key.startswith("encoder."))
    assert torch.equal(model.state_dict()[encoder_key], pretrained.state_dict()[encoder_key])

    z = torch.tensor([6, 1, 1], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]])
    prediction, dipoles = model(z, pos, torch.zeros(3, dtype=torch.long))
    assert prediction.shape == (1, 10)
    assert dipoles.shape == (1, 5, 3)
    expected_log_f = torch.log1p(
        OSCILLATOR_STRENGTH_FACTOR
        * prediction[:, 0::2].clamp_min(0)
        * dipoles.pow(2).sum(dim=-1)
    )
    assert torch.allclose(prediction[:, 1::2], expected_log_f)

    target = prediction.detach().clone()
    target_dipoles = -dipoles.detach()
    terms = qm9gwbse_loss(
        prediction,
        target,
        columns,
        predicted_dipoles=dipoles,
        target_dipoles=target_dipoles,
        dipole_weight=1.0,
        return_components=True,
    )
    assert terms["transition_dipole_phase_invariant_mse"].item() == pytest.approx(0.0, abs=1e-8)
    terms["total"].backward()
    assert model.state_queries.weight.grad is not None


def test_transition_dipole_sidecar_preserves_scalar_energy_model() -> None:
    pytest.importorskip("torch_geometric")
    torch = pytest.importorskip("torch")
    from gnn_excited.models.visnet import OSCILLATOR_STRENGTH_FACTOR, build_visnet, load_transfer_checkpoint

    columns = tuple(
        column
        for state in range(1, 6)
        for column in (f"S{state}_eV", f"log1p_S{state}_f")
    )
    kwargs = {
        "target_columns": columns,
        "hidden_channels": 32,
        "num_layers": 1,
        "num_rbf": 8,
        "cutoff": 3.0,
        "max_num_neighbors": 8,
    }
    scalar_model = build_visnet(**kwargs)
    sidecar = build_visnet(**kwargs, transition_dipole_sidecar=True)
    load_transfer_checkpoint(
        sidecar,
        {"model_state_dict": scalar_model.state_dict()},
        mode="frozen_sidecar",
    )
    assert all(
        not parameter.requires_grad
        for module in (sidecar.encoder, sidecar.energy_head, sidecar.oscillator_head)
        for parameter in module.parameters()
    )
    assert sidecar.state_queries.weight.requires_grad

    z = torch.tensor([6, 1, 1], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]])
    batch = torch.zeros(3, dtype=torch.long)
    scalar_prediction = scalar_model(z, pos, batch)
    prediction, dipoles = sidecar(z, pos, batch)
    assert torch.equal(prediction[:, 0::2], scalar_prediction[:, 0::2])
    expected_log_f = torch.log1p(
        OSCILLATOR_STRENGTH_FACTOR
        * prediction[:, 0::2].clamp_min(0)
        * dipoles.pow(2).sum(dim=-1)
    )
    assert torch.allclose(prediction[:, 1::2], expected_log_f)

    (prediction.sum() + dipoles.sum()).backward()
    assert all(
        parameter.grad is None
        for module in (sidecar.encoder, sidecar.energy_head, sidecar.oscillator_head)
        for parameter in module.parameters()
    )
    assert sidecar.state_queries.weight.grad is not None
