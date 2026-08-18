"""CPU-only smoke test for the model-side EvoEdit-OT components."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modules import BirthDeathUOT, ReportSlotEncoder, TextUOTTeacher, TransportConditionedEditor


def main() -> None:
    torch.manual_seed(42)
    batch_size, patches, tokens, dim = 2, 16, 20, 64
    history = torch.randn(batch_size, patches, dim, requires_grad=True)
    current = torch.randn(batch_size, patches, dim, requires_grad=True)
    report_tokens = torch.randn(batch_size, tokens, dim, requires_grad=True)
    report_mask = torch.ones(batch_size, tokens, dtype=torch.long)

    uot = BirthDeathUOT(dim=dim, num_queries=2, iterations=10, bottleneck_dim=32)
    slot_encoder = ReportSlotEncoder(dim=dim, num_slots=6, num_heads=4, bottleneck_dim=32)
    editor = TransportConditionedEditor(dim=dim, num_add_slots=2, attention_dim=32, bottleneck_dim=32)
    text_teacher = TextUOTTeacher(iterations=10)

    evidence = uot(current=current, history=history)
    prior_slots = slot_encoder(report_tokens, report_mask)
    edited = editor(prior_slots, evidence["edit_tokens"], new_token_count=2)
    target_slots = torch.randn(batch_size, 6, dim)
    loss = (
        evidence["uot_objective"].mean()
        + evidence["mass_regularization"].mean()
        + text_teacher.set_distance(edited["predicted_slots"], target_slots)
    )
    loss.backward()

    print("EvoEdit-OT module smoke test passed")
    print(f"edit_tokens={tuple(evidence['edit_tokens'].shape)}")
    print(f"predicted_slots={tuple(edited['predicted_slots'].shape)}")
    print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
