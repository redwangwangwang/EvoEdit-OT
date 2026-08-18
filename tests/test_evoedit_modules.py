from __future__ import annotations

import torch

from models.modules import (
    BirthDeathUOT,
    ReportSlotEncoder,
    TextUOTTeacher,
    TransportConditionedEditor,
    distribution_alignment_loss,
    invert_edit_distribution,
)


def test_birth_death_uot_shapes_and_gradients():
    torch.manual_seed(7)
    batch_size, patches, dim = 2, 16, 32
    history = torch.randn(batch_size, patches, dim, requires_grad=True)
    current = (history.detach() + 0.15 * torch.randn(batch_size, patches, dim)).requires_grad_(True)

    module = BirthDeathUOT(
        dim=dim,
        num_queries=3,
        epsilon=0.1,
        tau=0.8,
        iterations=12,
        spatial_weight=0.05,
        bottleneck_dim=16,
    )
    result = module(current=current, history=history)

    assert result["transport_plan"].shape == (batch_size, patches, patches)
    assert result["edit_tokens"].shape == (batch_size, 12, dim)
    assert result["edit_distribution"].shape == (batch_size, 4)
    assert torch.allclose(
        result["edit_distribution"].sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-5,
    )
    assert torch.isfinite(result["transport_plan"]).all()
    assert torch.isfinite(result["uot_objective"]).all()

    loss = (
        result["edit_tokens"].square().mean()
        + result["uot_objective"].mean()
        + result["mass_regularization"].mean()
    )
    loss.backward()
    assert history.grad is not None and torch.isfinite(history.grad).all()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert any(parameter.grad is not None for parameter in module.parameters())


def test_identical_inputs_are_keep_dominant():
    torch.manual_seed(11)
    features = torch.randn(2, 9, 24)
    module = BirthDeathUOT(
        dim=24,
        num_queries=2,
        epsilon=0.08,
        tau=1.5,
        iterations=20,
        spatial_weight=0.0,
        bottleneck_dim=12,
    )
    result = module(current=features, history=features)
    keep = result["raw_edit_distribution"][:, 3]
    modify = result["raw_edit_distribution"][:, 2]
    assert torch.all(keep > modify)


def test_temporal_distribution_inversion():
    distribution = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    inverted = invert_edit_distribution(distribution)
    assert torch.equal(inverted, torch.tensor([[0.2, 0.1, 0.3, 0.4]]))


def test_finding_slots_editor_and_text_teacher():
    torch.manual_seed(19)
    batch_size, length, dim = 2, 13, 32
    token_embeddings = torch.randn(batch_size, length, dim, requires_grad=True)
    attention_mask = torch.ones(batch_size, length, dtype=torch.long)
    attention_mask[1, 9:] = 0

    encoder = ReportSlotEncoder(
        dim=dim,
        num_slots=5,
        num_heads=4,
        dropout=0.0,
        bottleneck_dim=24,
    )
    editor = TransportConditionedEditor(
        dim=dim,
        num_add_slots=2,
        attention_dim=16,
        bottleneck_dim=24,
        dropout=0.0,
    )
    teacher = TextUOTTeacher(epsilon=0.1, tau=0.8, iterations=12)

    prior_slots = encoder(token_embeddings, attention_mask)
    edit_tokens = torch.randn(batch_size, 8, dim, requires_grad=True)
    edited = editor(prior_slots, edit_tokens, new_token_count=2)
    target_slots = torch.randn(batch_size, 5, dim)

    assert prior_slots.shape == (batch_size, 5, dim)
    assert edited["predicted_slots"].shape == (batch_size, 7, dim)
    assert edited["gate_probabilities"].shape == (batch_size, 5, 3)
    assert torch.allclose(
        edited["gate_probabilities"].sum(dim=-1),
        torch.ones(batch_size, 5),
        atol=1e-5,
    )

    teacher_output = teacher(prior_slots, target_slots)
    set_loss = teacher.set_distance(edited["predicted_slots"], target_slots)
    alignment = distribution_alignment_loss(
        teacher_output["edit_distribution"],
        teacher_output["edit_distribution"].detach(),
    )
    loss = set_loss + alignment + encoder.diversity_loss(prior_slots)
    loss.backward()

    assert torch.isfinite(loss)
    assert token_embeddings.grad is not None and torch.isfinite(token_embeddings.grad).all()
    assert edit_tokens.grad is not None and torch.isfinite(edit_tokens.grad).all()
