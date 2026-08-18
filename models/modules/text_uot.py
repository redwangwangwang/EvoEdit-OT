"""Text-side unbalanced transport used as an online edit teacher."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .birth_death_uot import _generalized_kl, unbalanced_log_sinkhorn


class TextUOTTeacher(nn.Module):
    """Derive soft report edits without sentence- or disease-level labels."""

    def __init__(self, epsilon: float = 0.10, tau: float = 0.8, iterations: int = 25) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.tau = float(tau)
        self.iterations = int(iterations)

    def _transport(self, source_slots: Tensor, target_slots: Tensor) -> Dict[str, Tensor]:
        if source_slots.ndim != 3 or target_slots.ndim != 3:
            raise ValueError("slot tensors must have shape [B,M,D]")
        if source_slots.shape[0] != target_slots.shape[0] or source_slots.shape[-1] != target_slots.shape[-1]:
            raise ValueError("source/target batch and embedding dimensions must match")

        source_normalized = F.normalize(source_slots, dim=-1)
        target_normalized = F.normalize(target_slots, dim=-1)
        cost = 1.0 - torch.einsum("bid,bjd->bij", source_normalized, target_normalized)
        cost = cost.clamp_min(0.0)

        batch_size, source_count, _ = source_slots.shape
        target_count = target_slots.shape[1]
        source_mass = torch.full(
            (batch_size, source_count),
            1.0 / max(1, source_count),
            device=source_slots.device,
            dtype=source_slots.dtype,
        )
        target_mass = torch.full(
            (batch_size, target_count),
            1.0 / max(1, target_count),
            device=target_slots.device,
            dtype=target_slots.dtype,
        )
        plan = unbalanced_log_sinkhorn(
            cost,
            source_mass,
            target_mass,
            epsilon=self.epsilon,
            tau=self.tau,
            iterations=self.iterations,
        )
        source_matched = plan.sum(dim=2)
        target_matched = plan.sum(dim=1)
        return {
            "cost": cost,
            "plan": plan,
            "source_mass": source_mass,
            "target_mass": target_mass,
            "source_matched": source_matched,
            "target_matched": target_matched,
        }

    def forward(self, prior_slots: Tensor, current_slots: Tensor) -> Dict[str, Tensor]:
        result = self._transport(prior_slots, current_slots)
        plan = result["plan"]
        cost = result["cost"]

        death = F.relu(result["source_mass"] - result["source_matched"]).sum(dim=-1)
        birth = F.relu(result["target_mass"] - result["target_matched"]).sum(dim=-1)
        modify = (plan * cost).sum(dim=(1, 2))
        keep = (plan * (1.0 - cost).clamp_min(0.0)).sum(dim=(1, 2))
        raw_distribution = torch.stack([birth, death, modify, keep], dim=-1)
        distribution = raw_distribution / raw_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        result.update(
            {
                "raw_edit_distribution": raw_distribution,
                "edit_distribution": distribution,
            }
        )
        return result

    def set_distance(self, predicted_slots: Tensor, target_slots: Tensor) -> Tensor:
        """Permutation-invariant UOT distance between two slot sets."""

        result = self._transport(predicted_slots, target_slots)
        plan = result["plan"]
        cost = result["cost"]
        transport = (plan * cost).sum(dim=(1, 2)) / plan.sum(dim=(1, 2)).clamp_min(1e-6)
        source_kl = _generalized_kl(result["source_matched"], result["source_mass"])
        target_kl = _generalized_kl(result["target_matched"], result["target_mass"])
        return (transport + self.tau * (source_kl + target_kl)).mean()


def distribution_alignment_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """KL divergence between normalized four-way edit distributions."""

    if student.shape != teacher.shape or student.shape[-1] != 4:
        raise ValueError("student and teacher must share shape [B,4]")
    student_log = student.clamp_min(1e-8).log()
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(student_log, teacher, reduction="batchmean")
