"""Latent finding slots and the transport-conditioned report editor."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ReportSlotEncoder(nn.Module):
    """Compress report-token embeddings into an unordered set of finding slots."""

    def __init__(
        self,
        dim: int,
        num_slots: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        bottleneck_dim: int = 512,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_slots = int(num_slots)
        self.slot_queries = nn.Parameter(torch.randn(num_slots, dim) / math.sqrt(dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden = min(max(64, bottleneck_dim), dim * 2)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must have shape [B,L,D]")
        batch_size = token_embeddings.shape[0]
        slots = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != token_embeddings.shape[:2]:
                raise ValueError("attention_mask must match [B,L]")
            # MultiheadAttention returns NaNs when every key is masked. Keep one
            # harmless padding embedding visible for empty reports.
            safe_mask = attention_mask.to(dtype=torch.bool).clone()
            empty = ~safe_mask.any(dim=1)
            if empty.any():
                safe_mask[empty, 0] = True
            key_padding_mask = ~safe_mask

        attended, _ = self.cross_attention(
            query=slots,
            key=token_embeddings,
            value=token_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        slots = self.norm1(slots + self.dropout(attended))
        slots = self.norm2(slots + self.dropout(self.ffn(slots)))
        return slots

    @staticmethod
    def diversity_loss(slots: Tensor) -> Tensor:
        """Penalize slot collapse while remaining permutation invariant."""

        if slots.ndim != 3:
            raise ValueError("slots must be [B,M,D]")
        normalized = F.normalize(slots, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        count = slots.shape[1]
        if count <= 1:
            return slots.new_zeros(())
        eye = torch.eye(count, device=slots.device, dtype=slots.dtype).unsqueeze(0)
        off_diagonal = similarity * (1.0 - eye)
        return off_diagonal.square().sum() / (slots.shape[0] * count * (count - 1))


class LowRankCrossAttention(nn.Module):
    """Cross attention whose values stay in the original model dimension."""

    def __init__(self, dim: int, attention_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        attention_dim = min(attention_dim, dim)
        self.query_proj = nn.Linear(dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(dim, attention_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = attention_dim ** -0.5

    def forward(self, queries: Tensor, context: Tensor, context_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        q = self.query_proj(queries)
        k = self.key_proj(context)
        logits = torch.matmul(q, k.transpose(1, 2)) * self.scale
        if context_mask is not None:
            logits = logits.masked_fill(~context_mask[:, None, :].bool(), torch.finfo(logits.dtype).min)
        attention = self.dropout(logits.softmax(dim=-1))
        values = torch.matmul(attention, context)
        return values, attention


class TransportConditionedEditor(nn.Module):
    """Apply KEEP/DELETE/MODIFY/ADD operations to prior finding slots."""

    def __init__(
        self,
        dim: int,
        num_add_slots: int = 4,
        attention_dim: int = 256,
        bottleneck_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_add_slots = int(num_add_slots)
        self.slot_attention = LowRankCrossAttention(dim, attention_dim=attention_dim, dropout=dropout)
        self.add_attention = LowRankCrossAttention(dim, attention_dim=attention_dim, dropout=dropout)
        self.add_queries = nn.Parameter(torch.randn(num_add_slots, dim) / math.sqrt(dim))

        hidden = min(max(64, bottleneck_dim), dim * 2)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),  # keep, delete, modify
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.slot_norm = nn.LayerNorm(dim)
        self.add_norm = nn.LayerNorm(dim)

    def forward(
        self,
        prior_slots: Tensor,
        edit_tokens: Tensor,
        *,
        new_token_count: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if prior_slots.ndim != 3 or edit_tokens.ndim != 3:
            raise ValueError("prior_slots and edit_tokens must be [B,N,D]")
        if prior_slots.shape[0] != edit_tokens.shape[0] or prior_slots.shape[-1] != edit_tokens.shape[-1]:
            raise ValueError("prior_slots/edit_tokens batch and embedding dimensions must match")

        edit_context, edit_attention = self.slot_attention(prior_slots, edit_tokens)
        fused = torch.cat([prior_slots, edit_context], dim=-1)
        gate_logits = self.gate(fused)
        gate_probabilities = gate_logits.softmax(dim=-1)
        keep_probability = gate_probabilities[..., 0:1]
        delete_probability = gate_probabilities[..., 1:2]
        modify_probability = gate_probabilities[..., 2:3]

        modified_slots = self.slot_norm(prior_slots + self.delta(fused))
        edited_prior = keep_probability * prior_slots + modify_probability * modified_slots
        edited_prior = self.slot_norm(edited_prior)

        new_context = edit_tokens
        if new_token_count is not None:
            new_context = edit_tokens[:, : max(1, int(new_token_count))]
        add_queries = self.add_queries.unsqueeze(0).expand(prior_slots.shape[0], -1, -1)
        add_context, add_attention = self.add_attention(add_queries, new_context)
        add_slots = self.add_norm(add_queries + add_context)

        predicted_slots = torch.cat([edited_prior, add_slots], dim=1)
        return {
            "predicted_slots": predicted_slots,
            "edited_prior_slots": edited_prior,
            "add_slots": add_slots,
            "gate_logits": gate_logits,
            "gate_probabilities": gate_probabilities,
            "delete_probability": delete_probability,
            "edit_attention": edit_attention,
            "add_attention": add_attention,
        }
