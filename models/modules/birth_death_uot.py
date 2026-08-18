"""Differentiable birth--death unbalanced optimal transport.

This module replaces BiOTPrompt's balanced, thresholded Sinkhorn detector with a
fully differentiable unbalanced transport layer.  Unmatched target mass is
interpreted as *birth* (new evidence), unmatched source mass as *death*
(resolved evidence), and matched-but-dissimilar mass as modification.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _generalized_kl(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Generalized KL divergence for non-normalized positive measures."""

    x = x.clamp_min(eps)
    y = y.clamp_min(eps)
    return (x * (x.log() - y.log()) - x + y).sum(dim=-1)


def unbalanced_log_sinkhorn(
    cost: Tensor,
    source_mass: Tensor,
    target_mass: Tensor,
    *,
    epsilon: float = 0.08,
    tau: float = 0.8,
    iterations: int = 30,
    clamp_log: float = 50.0,
) -> Tensor:
    """Compute an entropic unbalanced transport plan in the log domain.

    Args:
        cost: Pairwise cost with shape ``[B, N_source, N_target]``.
        source_mass: Positive source measure ``[B, N_source]``.
        target_mass: Positive target measure ``[B, N_target]``.
        epsilon: Entropic regularization strength.
        tau: Marginal-relaxation strength. Larger values approach balanced OT.
        iterations: Number of generalized Sinkhorn iterations.
        clamp_log: Numerical clamp applied before exponentiating the plan.

    Returns:
        A non-negative transport plan with shape ``[B, N_source, N_target]``.
    """

    if cost.ndim != 3:
        raise ValueError(f"cost must be [B, Ns, Nt], got {tuple(cost.shape)}")
    if source_mass.shape != cost.shape[:2]:
        raise ValueError("source_mass shape must match cost's source dimensions")
    if target_mass.shape != (cost.shape[0], cost.shape[2]):
        raise ValueError("target_mass shape must match cost's target dimensions")
    if epsilon <= 0 or tau <= 0:
        raise ValueError("epsilon and tau must be positive")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    # OT scaling is especially sensitive in fp16/bfloat16.  Run the fixed-point
    # iterations in float32 while preserving a differentiable cast back to the
    # model dtype.
    output_dtype = cost.dtype
    work_dtype = torch.float32 if cost.dtype in {torch.float16, torch.bfloat16} else cost.dtype
    work_cost = cost.to(work_dtype)
    work_source_mass = source_mass.to(work_dtype)
    work_target_mass = target_mass.to(work_dtype)

    eps = torch.finfo(work_dtype).eps
    log_a = work_source_mass.clamp_min(eps).log()
    log_b = work_target_mass.clamp_min(eps).log()
    log_kernel = -work_cost / epsilon

    # Generalized Sinkhorn exponent for KL-relaxed marginals.
    relaxation = tau / (tau + epsilon)
    log_u = torch.zeros_like(work_source_mass)
    log_v = torch.zeros_like(work_target_mass)

    for _ in range(iterations):
        source_lse = torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
        log_u = relaxation * (log_a - source_lse)
        target_lse = torch.logsumexp(log_kernel + log_u.unsqueeze(2), dim=1)
        log_v = relaxation * (log_b - target_lse)

    log_plan = log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1)
    plan = torch.exp(log_plan.clamp(min=-clamp_log, max=clamp_log))
    return plan.to(output_dtype)


class PatchMassPredictor(nn.Module):
    """Predict a positive local measure and a sample-level total-mass scale."""

    def __init__(self, dim: int, bottleneck_dim: int = 256, min_scale: float = 0.5, max_scale: float = 1.5):
        super().__init__()
        if min_scale <= 0 or max_scale <= min_scale:
            raise ValueError("Expected 0 < min_scale < max_scale")
        hidden = min(max(32, bottleneck_dim), dim)
        self.local_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B, N, D]")
        local = F.softplus(self.local_head(features).squeeze(-1)) + 1e-6
        local = local / local.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = features.mean(dim=1)
        scale = self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(
            self.global_head(pooled).squeeze(-1)
        )
        return local * scale.unsqueeze(-1)


class WeightedQueryPool(nn.Module):
    """Pool a variable patch set into a fixed number of continuous tokens."""

    def __init__(self, dim: int, num_queries: int, attention_dim: int = 256):
        super().__init__()
        self.num_queries = int(num_queries)
        attention_dim = min(attention_dim, dim)
        self.key_proj = nn.Linear(dim, attention_dim, bias=False)
        self.queries = nn.Parameter(torch.randn(num_queries, attention_dim) / math.sqrt(attention_dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, values: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor]:
        if values.ndim != 3 or weights.ndim != 2:
            raise ValueError("values must be [B,N,D] and weights must be [B,N]")
        if values.shape[:2] != weights.shape:
            raise ValueError("weights must match the first two value dimensions")

        keys = F.normalize(self.key_proj(values), dim=-1)
        queries = F.normalize(self.queries, dim=-1)
        logits = torch.einsum("qd,bnd->bqn", queries, keys) / math.sqrt(keys.shape[-1])
        logits = logits + weights.clamp_min(1e-8).log().unsqueeze(1)
        attention = logits.softmax(dim=-1)
        pooled = torch.einsum("bqn,bnd->bqd", attention, values)

        # Exactly-zero evidence should not fabricate a strong content token.
        valid = (weights.sum(dim=-1, keepdim=True) > 1e-7).to(values.dtype)
        pooled = self.output_norm(pooled) * valid.unsqueeze(-1)
        return pooled, attention


class BirthDeathUOT(nn.Module):
    """Birth--death UOT with differentiable evolution-token extraction."""

    EDIT_TYPES = ("new", "resolved", "modify", "keep")

    def __init__(
        self,
        dim: int,
        num_queries: int = 4,
        epsilon: float = 0.08,
        tau: float = 0.8,
        iterations: int = 30,
        spatial_weight: float = 0.10,
        bottleneck_dim: int = 256,
        minimum_matched_fraction: float = 0.15,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.epsilon = float(epsilon)
        self.tau = float(tau)
        self.iterations = int(iterations)
        self.spatial_weight = float(spatial_weight)
        self.minimum_matched_fraction = float(minimum_matched_fraction)

        self.mass_predictor = PatchMassPredictor(dim, bottleneck_dim=bottleneck_dim)
        self.poolers = nn.ModuleDict(
            {name: WeightedQueryPool(dim, num_queries, attention_dim=bottleneck_dim) for name in self.EDIT_TYPES}
        )
        self.type_embeddings = nn.Parameter(torch.randn(len(self.EDIT_TYPES), 1, dim) / math.sqrt(dim))
        self.edit_norm = nn.LayerNorm(dim)

    @staticmethod
    def _positions(length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        side = int(round(math.sqrt(length)))
        if side * side == length:
            y, x = torch.meshgrid(
                torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
                torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
                indexing="ij",
            )
            return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
        x = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
        return torch.stack([x, torch.zeros_like(x)], dim=-1)

    def _cost(self, history: Tensor, current: Tensor) -> Tensor:
        history_normalized = F.normalize(history, dim=-1)
        current_normalized = F.normalize(current, dim=-1)
        semantic_cost = 1.0 - torch.einsum("bid,bjd->bij", history_normalized, current_normalized)

        history_pos = self._positions(history.shape[1], device=history.device, dtype=history.dtype)
        current_pos = self._positions(current.shape[1], device=current.device, dtype=current.dtype)
        spatial_cost = torch.cdist(history_pos, current_pos, p=1).unsqueeze(0)
        return semantic_cost.clamp_min(0.0) + self.spatial_weight * spatial_cost

    def forward(self, current: Tensor, history: Tensor) -> Dict[str, Tensor]:
        if current.ndim != 3 or history.ndim != 3:
            raise ValueError("current and history must have shape [B,N,D]")
        if current.shape[0] != history.shape[0] or current.shape[-1] != history.shape[-1]:
            raise ValueError("current/history batch and embedding dimensions must match")
        if current.shape[-1] != self.dim:
            raise ValueError(f"Expected embedding dimension {self.dim}, got {current.shape[-1]}")

        history_mass = self.mass_predictor(history)
        current_mass = self.mass_predictor(current)
        cost = self._cost(history, current)
        plan = unbalanced_log_sinkhorn(
            cost,
            history_mass,
            current_mass,
            epsilon=self.epsilon,
            tau=self.tau,
            iterations=self.iterations,
        )

        history_matched = plan.sum(dim=2)
        current_matched = plan.sum(dim=1)
        death_mass = F.relu(history_mass - history_matched)
        birth_mass = F.relu(current_mass - current_matched)

        barycentric_history = torch.einsum("bij,bid->bjd", plan, history)
        barycentric_history = barycentric_history / current_matched.unsqueeze(-1).clamp_min(1e-6)
        delta = current - barycentric_history
        similarity = F.cosine_similarity(current, barycentric_history, dim=-1).clamp(-1.0, 1.0)
        modify_mass = current_matched * (1.0 - similarity) * 0.5
        keep_mass = current_matched * (1.0 + similarity) * 0.5

        features = {
            "new": current,
            "resolved": history,
            "modify": delta,
            "keep": 0.5 * (current + barycentric_history),
        }
        weights = {
            "new": birth_mass,
            "resolved": death_mass,
            "modify": modify_mass,
            "keep": keep_mass,
        }

        token_groups = []
        pool_attentions = {}
        for index, name in enumerate(self.EDIT_TYPES):
            pooled, attention = self.poolers[name](features[name], weights[name])
            token_groups.append(pooled + self.type_embeddings[index])
            pool_attentions[name] = attention
        edit_tokens = self.edit_norm(torch.cat(token_groups, dim=1))

        raw_distribution = torch.stack(
            [birth_mass.sum(-1), death_mass.sum(-1), modify_mass.sum(-1), keep_mass.sum(-1)], dim=-1
        )
        edit_distribution = raw_distribution / raw_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        transport_cost = (plan * cost).sum(dim=(1, 2))
        entropy = -(plan.clamp_min(1e-8) * plan.clamp_min(1e-8).log()).sum(dim=(1, 2))
        source_kl = _generalized_kl(history_matched, history_mass)
        target_kl = _generalized_kl(current_matched, current_mass)
        uot_objective = transport_cost - self.epsilon * entropy + self.tau * (source_kl + target_kl)

        average_total_mass = 0.5 * (history_mass.sum(-1) + current_mass.sum(-1))
        matched_fraction = plan.sum(dim=(1, 2)) / average_total_mass.clamp_min(1e-6)
        mass_regularization = (
            (history_mass.sum(-1) - 1.0).square()
            + (current_mass.sum(-1) - 1.0).square()
            + F.relu(self.minimum_matched_fraction - matched_fraction).square()
        )

        return {
            "transport_plan": plan,
            "cost": cost,
            "history_mass": history_mass,
            "current_mass": current_mass,
            "birth_mass": birth_mass,
            "death_mass": death_mass,
            "modify_mass": modify_mass,
            "keep_mass": keep_mass,
            "barycentric_history": barycentric_history,
            "edit_tokens": edit_tokens,
            "raw_edit_distribution": raw_distribution,
            "edit_distribution": edit_distribution,
            "transport_cost": transport_cost,
            "uot_objective": uot_objective,
            "mass_regularization": mass_regularization,
            "matched_fraction": matched_fraction,
            "pool_attention_new": pool_attentions["new"],
            "pool_attention_resolved": pool_attentions["resolved"],
            "pool_attention_modify": pool_attentions["modify"],
            "pool_attention_keep": pool_attentions["keep"],
        }


def invert_edit_distribution(distribution: Tensor) -> Tensor:
    """Map [new, resolved, modify, keep] to its time-reversed ordering."""

    if distribution.shape[-1] != 4:
        raise ValueError("Expected a four-way edit distribution")
    return distribution[..., [1, 0, 2, 3]]
