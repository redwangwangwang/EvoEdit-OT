"""EvoEdit-OT integration for the BiOTPrompt longitudinal RRG backbone.

The original repository is kept as a pinned git submodule in ``upstream/``.
This file subclasses its Lightning model and replaces hard patch-index prompts
with birth--death UOT evidence, latent finding slots, and an explicit report
editor.  No dataset fields or splits are changed.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modules import (
    BirthDeathUOT,
    ReportSlotEncoder,
    TextUOTTeacher,
    TransportConditionedEditor,
    distribution_alignment_loss,
    invert_edit_distribution,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_UPSTREAM_ROOT = _REPOSITORY_ROOT / "upstream"
_UPSTREAM_MODEL = _UPSTREAM_ROOT / "models" / "R2GenGPT.py"

if not _UPSTREAM_MODEL.exists():
    raise RuntimeError(
        "BiOTPrompt submodule is missing. Run `git submodule update --init --recursive` "
        "or clone this repository with `--recurse-submodules`."
    )

# The upstream model imports evalcap/lightning_tools as top-level packages.
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

_spec = importlib.util.spec_from_file_location("evoedit_biotprompt_upstream", _UPSTREAM_MODEL)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load upstream model from {_UPSTREAM_MODEL}")
_upstream_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_upstream_module)
_BiOTPromptR2GenGPT = _upstream_module.R2GenGPT


def _get_arg(args, name: str, default):
    return getattr(args, name, default)


def _compatible_num_heads(dim: int, requested: int) -> int:
    requested = max(1, min(int(requested), dim))
    while dim % requested != 0 and requested > 1:
        requested -= 1
    return requested


class _Classifier(nn.Module):
    def __init__(self, dim: int, num_labels: int = 14, num_states: int = 4) -> None:
        super().__init__()
        hidden = min(512, dim)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, num_labels * num_states))
        self.num_labels = num_labels
        self.num_states = num_states

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).view(-1, self.num_states, self.num_labels)


class R2GenGPT(_BiOTPromptR2GenGPT):
    """BiOTPrompt backbone augmented with the EvoEdit-OT method."""

    def __init__(self, args) -> None:
        # Upstream loads delta checkpoints before subclass modules exist. Delay
        # checkpoint loading until the complete EvoEdit-OT graph is initialized.
        delta_file = _get_arg(args, "delta_file", None)
        setattr(args, "delta_file", None)
        super().__init__(args)
        setattr(args, "delta_file", delta_file)
        self.hparams.delta_file = delta_file

        hidden_size = int(self.llama_model.config.hidden_size)
        requested_heads = _get_arg(args, "evoedit_num_heads", 8)
        num_heads = _compatible_num_heads(hidden_size, requested_heads)
        bottleneck_dim = int(_get_arg(args, "evoedit_bottleneck_dim", 512))

        self.num_edit_queries = int(_get_arg(args, "evoedit_num_edit_queries", 4))
        self.prior_max_length = int(_get_arg(args, "evoedit_prior_max_length", 96))
        self.detach_text_teacher = bool(_get_arg(args, "evoedit_detach_text_teacher", True))

        self.evolution_encoder = BirthDeathUOT(
            dim=hidden_size,
            num_queries=self.num_edit_queries,
            epsilon=float(_get_arg(args, "evoedit_uot_epsilon", 0.08)),
            tau=float(_get_arg(args, "evoedit_uot_tau", 0.8)),
            iterations=int(_get_arg(args, "evoedit_uot_iterations", 30)),
            spatial_weight=float(_get_arg(args, "evoedit_spatial_weight", 0.10)),
            bottleneck_dim=min(256, bottleneck_dim),
            minimum_matched_fraction=float(_get_arg(args, "evoedit_min_matched_fraction", 0.15)),
        )
        self.finding_encoder = ReportSlotEncoder(
            dim=hidden_size,
            num_slots=int(_get_arg(args, "evoedit_num_finding_slots", 8)),
            num_heads=num_heads,
            dropout=float(_get_arg(args, "evoedit_dropout", 0.1)),
            bottleneck_dim=bottleneck_dim,
        )
        self.report_editor = TransportConditionedEditor(
            dim=hidden_size,
            num_add_slots=int(_get_arg(args, "evoedit_num_add_slots", 4)),
            attention_dim=min(256, bottleneck_dim),
            bottleneck_dim=bottleneck_dim,
            dropout=float(_get_arg(args, "evoedit_dropout", 0.1)),
        )
        self.text_uot_teacher = TextUOTTeacher(
            epsilon=float(_get_arg(args, "evoedit_text_uot_epsilon", 0.10)),
            tau=float(_get_arg(args, "evoedit_text_uot_tau", 0.8)),
            iterations=int(_get_arg(args, "evoedit_text_uot_iterations", 25)),
        )

        # Make the classifier compatible with non-4096 hidden-size LLMs.
        self.cls_head_visual = _Classifier(hidden_size)
        self.cls_head_text = _Classifier(hidden_size)
        self.criterion_cls = nn.CrossEntropyLoss()

        # The thresholded detector is deliberately no longer used.
        self.detector = None

        if delta_file is not None:
            checkpoint = torch.load(delta_file, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(
                f"Loaded checkpoint from {delta_file}; "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    # ------------------------------------------------------------------
    # Encoding and prompt construction
    # ------------------------------------------------------------------
    def encode_img(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Run the vision encoder once (the upstream code runs it twice)."""

        outputs = self.visual_encoder(images)
        image_embed = outputs["last_hidden_state"]
        image_embed_pooler = outputs["pooler_output"]
        inputs_llama = self.llama_proj(image_embed)
        inputs_llama_pooler = self.llama_proj(image_embed_pooler)
        attention = torch.ones(inputs_llama.shape[:2], dtype=torch.long, device=images.device)
        return inputs_llama, inputs_llama_pooler, attention

    def _tokenize_reports(
        self,
        reports: Sequence[str],
        *,
        device: torch.device,
        max_length: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        self.llama_tokenizer.padding_side = "right"
        tokenized = self.llama_tokenizer(
            list(reports),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(device)
        embeddings = self.embed_tokens(tokenized.input_ids)
        return tokenized.input_ids, embeddings, tokenized.attention_mask

    def _encode_report_slots(
        self,
        reports: Sequence[str],
        *,
        device: torch.device,
        max_length: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        _, token_embeddings, attention_mask = self._tokenize_reports(
            reports, device=device, max_length=max_length
        )
        slots = self.finding_encoder(token_embeddings, attention_mask)
        return slots, token_embeddings, attention_mask

    def _embed_fragment(self, text: str, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        tokenized = self.llama_tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        embeddings = self.embed_tokens(tokenized.input_ids).expand(batch_size, -1, -1)
        attention = tokenized.attention_mask.expand(batch_size, -1)
        return embeddings, attention

    @staticmethod
    def _ones_for(embeddings: Tensor) -> Tensor:
        return torch.ones(embeddings.shape[:2], dtype=torch.long, device=embeddings.device)

    def prompt_wrap(
        self,
        current_image: Tensor,
        history_image: Tensor,
        edit_tokens: Tensor,
        prior_report_embeddings: Tensor,
        prior_report_attention: Tensor,
        predicted_slots: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Build a continuous, end-to-end differentiable longitudinal prompt."""

        batch_size = current_image.shape[0]
        device = current_image.device
        fragments = [
            "Human: <CurrentImage>",
            "</CurrentImage>\nGenerate the current chest X-ray report by minimally and clinically editing "
            "the prior findings. Use the evolution evidence below; do not copy resolved abnormalities.\n"
            "<EvolutionEvidence>",
            "</EvolutionEvidence>\n<PriorImage>",
            "</PriorImage>\n<PriorReport>",
            "</PriorReport>\n<EditedFindingSlots>",
            "</EditedFindingSlots>\nAssistant:",
        ]
        fragment_embeddings = [self._embed_fragment(text, batch_size, device) for text in fragments]

        parts = [
            fragment_embeddings[0],
            (current_image, self._ones_for(current_image)),
            fragment_embeddings[1],
            (edit_tokens, self._ones_for(edit_tokens)),
            fragment_embeddings[2],
            (history_image, self._ones_for(history_image)),
            fragment_embeddings[3],
            (prior_report_embeddings, prior_report_attention),
            fragment_embeddings[4],
            (predicted_slots, self._ones_for(predicted_slots)),
            fragment_embeddings[5],
        ]
        embeddings = torch.cat([part[0] for part in parts], dim=1)
        attention = torch.cat([part[1] for part in parts], dim=1)
        return embeddings, attention

    def _prepare_evolution_prompt(self, samples: Dict[str, Tensor]) -> Dict[str, Tensor]:
        image = samples["image"]
        current_raw, current_pooler, _ = self.encode_img(image)
        current = self.layer_norm(current_raw)

        history_image = samples["context_image"]
        history_raw, history_pooler, _ = self.encode_img(history_image)
        history = self.layer_norm(history_raw)

        evidence = self.evolution_encoder(current=current, history=history)
        prior_reports = samples["context_input_text"]
        prior_slots, prior_token_embeddings, prior_attention = self._encode_report_slots(
            prior_reports,
            device=image.device,
            max_length=self.prior_max_length,
        )
        editor_output = self.report_editor(
            prior_slots,
            evidence["edit_tokens"],
            new_token_count=self.num_edit_queries,
        )
        prompt_embeddings, prompt_attention = self.prompt_wrap(
            current,
            history,
            evidence["edit_tokens"],
            prior_token_embeddings,
            prior_attention,
            editor_output["predicted_slots"],
        )
        return {
            "current": current,
            "history": history,
            "current_pooler": current_pooler,
            "history_pooler": history_pooler,
            "prior_slots": prior_slots,
            "prior_token_embeddings": prior_token_embeddings,
            "prior_attention": prior_attention,
            "evidence": evidence,
            "editor_output": editor_output,
            "prompt_embeddings": prompt_embeddings,
            "prompt_attention": prompt_attention,
        }

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------
    def _classification_losses(
        self,
        outputs,
        target_attention: Tensor,
        current_pooler: Tensor,
        current_labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        target_length = target_attention.shape[1]
        hidden = outputs.hidden_states[-1][:, -target_length:, :]
        mask = target_attention.unsqueeze(-1).to(hidden.dtype)
        text_embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        text_embedding = self.layer_norm(text_embedding)

        text_logits = self.cls_head_text(text_embedding)
        image_logits = self.cls_head_visual(current_pooler)
        labels = current_labels[:, :14]
        text_loss = self.criterion_cls(text_logits, labels)
        image_loss = self.criterion_cls(image_logits, labels)

        text_log_probability = F.log_softmax(text_logits.permute(0, 2, 1), dim=-1)
        image_probability = F.softmax(image_logits.permute(0, 2, 1).detach(), dim=-1)
        consistency = F.kl_div(text_log_probability, image_probability, reduction="batchmean")
        return text_loss, image_loss, consistency

    def forward(self, samples: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prepared = self._prepare_evolution_prompt(samples)
        image = samples["image"]

        target_text = [text + self.end_sym for text in samples["input_text"]]
        target_ids, target_embeddings, target_attention = self._tokenize_reports(
            target_text,
            device=image.device,
            max_length=int(self.hparams.max_length),
        )
        targets = target_ids.masked_fill(target_attention == 0, -100)

        prompt_embeddings = prepared["prompt_embeddings"]
        prompt_attention = prepared["prompt_attention"]
        batch_size = prompt_embeddings.shape[0]
        bos = torch.full(
            (batch_size, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=target_ids.dtype,
            device=image.device,
        )
        bos_embeddings = self.embed_tokens(bos)
        bos_attention = torch.ones((batch_size, 1), dtype=prompt_attention.dtype, device=image.device)

        empty_targets = torch.full(
            (batch_size, prompt_embeddings.shape[1] + 1),
            fill_value=-100,
            dtype=torch.long,
            device=image.device,
        )
        labels = torch.cat([empty_targets, targets], dim=1)
        inputs_embeds = torch.cat([bos_embeddings, prompt_embeddings, target_embeddings], dim=1)
        attention_mask = torch.cat([bos_attention, prompt_attention, target_attention], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=labels,
        )

        zero = outputs.loss.new_zeros(())
        text_cls_loss = image_cls_loss = vl_consistency_loss = zero
        if "current_labels" in samples:
            text_cls_loss, image_cls_loss, vl_consistency_loss = self._classification_losses(
                outputs,
                target_attention,
                prepared["current_pooler"],
                samples["current_labels"],
            )

        current_slots, _, _ = self._encode_report_slots(
            samples["input_text"],
            device=image.device,
            max_length=self.prior_max_length,
        )
        target_slots = current_slots.detach() if self.detach_text_teacher else current_slots

        slot_loss = self.text_uot_teacher.set_distance(
            prepared["editor_output"]["predicted_slots"],
            target_slots,
        )
        teacher_prior = prepared["prior_slots"].detach() if self.detach_text_teacher else prepared["prior_slots"]
        text_teacher = self.text_uot_teacher(teacher_prior, target_slots)
        edit_alignment_loss = distribution_alignment_loss(
            prepared["evidence"]["edit_distribution"],
            text_teacher["edit_distribution"].detach(),
        )

        diversity_loss = 0.5 * (
            self.finding_encoder.diversity_loss(prepared["prior_slots"])
            + self.finding_encoder.diversity_loss(current_slots)
        )
        transport_loss = prepared["evidence"]["uot_objective"].mean()
        mass_loss = prepared["evidence"]["mass_regularization"].mean()

        inversion_loss = zero
        reverse_slot_loss = zero
        inversion_weight = float(_get_arg(self.hparams, "evoedit_inversion_weight", 0.10))
        reverse_weight = float(_get_arg(self.hparams, "evoedit_reverse_weight", 0.20))
        if inversion_weight > 0.0 or reverse_weight > 0.0:
            reverse_evidence = self.evolution_encoder(
                current=prepared["history"],
                history=prepared["current"],
            )
            inversion_loss = F.smooth_l1_loss(
                prepared["evidence"]["edit_distribution"],
                invert_edit_distribution(reverse_evidence["edit_distribution"]),
            )
            if reverse_weight > 0.0:
                reverse_editor = self.report_editor(
                    target_slots,
                    reverse_evidence["edit_tokens"],
                    new_token_count=self.num_edit_queries,
                )
                reverse_slot_loss = self.text_uot_teacher.set_distance(
                    reverse_editor["predicted_slots"],
                    prepared["prior_slots"].detach(),
                )

        total_loss = (
            outputs.loss
            + float(_get_arg(self.hparams, "evoedit_vl_weight", 1.0))
            * (vl_consistency_loss + text_cls_loss + image_cls_loss)
            + float(_get_arg(self.hparams, "evoedit_slot_weight", 0.50)) * slot_loss
            + float(_get_arg(self.hparams, "evoedit_edit_align_weight", 0.20)) * edit_alignment_loss
            + float(_get_arg(self.hparams, "evoedit_transport_weight", 0.05)) * transport_loss
            + float(_get_arg(self.hparams, "evoedit_mass_weight", 0.05)) * mass_loss
            + float(_get_arg(self.hparams, "evoedit_diversity_weight", 0.01)) * diversity_loss
            + inversion_weight * inversion_loss
            + reverse_weight * reverse_slot_loss
        )

        return {
            "loss": total_loss,
            "loss_lm": outputs.loss.detach(),
            "loss_slot": slot_loss.detach(),
            "loss_edit_align": edit_alignment_loss.detach(),
            "loss_transport": transport_loss.detach(),
            "loss_mass": mass_loss.detach(),
            "loss_inversion": inversion_loss.detach(),
            "loss_reverse": reverse_slot_loss.detach(),
            "matched_fraction": prepared["evidence"]["matched_fraction"].mean().detach(),
        }

    # ------------------------------------------------------------------
    # Validation and testing
    # ------------------------------------------------------------------
    def _generation_step(self, samples: Dict[str, Tensor], stage: str):
        image = samples["image"]
        target_ids, _, _ = self._tokenize_reports(
            samples["input_text"],
            device=image.device,
            max_length=int(self.hparams.max_length),
        )
        prepared = self._prepare_evolution_prompt(samples)
        prompt_embeddings = prepared["prompt_embeddings"]
        prompt_attention = prepared["prompt_attention"]

        batch_size = prompt_embeddings.shape[0]
        bos = torch.full(
            (batch_size, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=image.device,
        )
        bos_embeddings = self.embed_tokens(bos)
        bos_attention = torch.ones((batch_size, 1), dtype=prompt_attention.dtype, device=image.device)
        inputs_embeds = torch.cat([bos_embeddings, prompt_embeddings], dim=1)
        attention_mask = torch.cat([bos_attention, prompt_attention], dim=1)

        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "pad_token_id": self.llama_tokenizer.pad_token_id,
            "num_beams": int(self.hparams.beam_size),
            "do_sample": bool(self.hparams.do_sample),
            "min_new_tokens": int(self.hparams.min_new_tokens),
            "max_new_tokens": int(self.hparams.max_new_tokens),
            "repetition_penalty": float(self.hparams.repetition_penalty),
            "length_penalty": float(self.hparams.length_penalty),
            "no_repeat_ngram_size": int(_get_arg(self.hparams, "no_repeat_ngram_size", 0)),
        }
        temperature = float(_get_arg(self.hparams, "temperature", 0.0))
        if generation_kwargs["do_sample"] and temperature > 0.0:
            generation_kwargs["temperature"] = temperature

        generated_tokens = self.llama_model.generate(**generation_kwargs)
        hypotheses = [self.decode(tokens) for tokens in generated_tokens]
        references = [self.decode(tokens) for tokens in target_ids]
        output = {"hypo": hypotheses, "ref": references, "id": samples["id"]}
        if stage == "val":
            self.val_step_outputs.append(output)
        elif stage == "test":
            self.test_step_outputs.append(output)
        else:
            raise ValueError(f"Unknown generation stage: {stage}")
        return hypotheses, references

    def validation_step(self, samples, batch_idx):
        return self._generation_step(samples, "val")

    def test_step(self, samples, batch_idx):
        return self._generation_step(samples, "test")
