"""Command-line configuration for EvoEdit-OT.

All original BiOTPrompt arguments are retained; the ``evoedit_*`` options only
control model-side behavior and require no annotation or split changes.
"""

from __future__ import annotations

import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


parser = argparse.ArgumentParser(description="Hyper-parameters for EvoEdit-OT")

# Dataset (unchanged from BiOTPrompt)
parser.add_argument("--test", action="store_true", help="only run the test set")
parser.add_argument("--validate", action="store_true", help="only run the validation set")
parser.add_argument("--dataset", type=str, default="iu-xray", help="iu-xray or mimic-cxr")
parser.add_argument("--annotation", type=str, default="./data/iu_xray/data.json")
parser.add_argument("--base_dir", type=str, default="./data/iu_xray/images")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--val_batch_size", default=12, type=int)
parser.add_argument("--test_batch_size", default=16, type=int)
parser.add_argument("--prefetch_factor", default=4, type=int)
parser.add_argument("--num_workers", default=4, type=int)

# Backbone
parser.add_argument("--vision_model", default="./data/swin-base-patch4-window7-224", type=str)
parser.add_argument("--llama_model", default="./data/Llama-2-7b-chat-hf/", type=str)
parser.add_argument("--freeze_vm", default=True, type=str2bool)
parser.add_argument("--llm_use_lora", default=False, type=str2bool)
parser.add_argument("--llm_r", default=16, type=int)
parser.add_argument("--llm_alpha", default=16, type=int)
parser.add_argument("--vis_use_lora", default=True, type=str2bool)
parser.add_argument("--vis_r", default=16, type=int)
parser.add_argument("--vis_alpha", default=16, type=int)
parser.add_argument("--lora_dropout", default=0.1, type=float)
parser.add_argument("--global_only", default=False, type=str2bool)
parser.add_argument("--low_resource", default=False, type=str2bool)
parser.add_argument("--end_sym", default="</s>", type=str)

# EvoEdit-OT architecture
parser.add_argument("--evoedit_num_edit_queries", default=4, type=int)
parser.add_argument("--evoedit_num_finding_slots", default=8, type=int)
parser.add_argument("--evoedit_num_add_slots", default=4, type=int)
parser.add_argument("--evoedit_num_heads", default=8, type=int)
parser.add_argument("--evoedit_bottleneck_dim", default=512, type=int)
parser.add_argument("--evoedit_dropout", default=0.1, type=float)
parser.add_argument("--evoedit_prior_max_length", default=96, type=int)
parser.add_argument("--evoedit_detach_text_teacher", default=True, type=str2bool)

# Birth--death visual UOT
parser.add_argument("--evoedit_uot_epsilon", default=0.08, type=float)
parser.add_argument("--evoedit_uot_tau", default=0.8, type=float)
parser.add_argument("--evoedit_uot_iterations", default=30, type=int)
parser.add_argument("--evoedit_spatial_weight", default=0.10, type=float)
parser.add_argument("--evoedit_min_matched_fraction", default=0.15, type=float)

# Text-UOT teacher
parser.add_argument("--evoedit_text_uot_epsilon", default=0.10, type=float)
parser.add_argument("--evoedit_text_uot_tau", default=0.8, type=float)
parser.add_argument("--evoedit_text_uot_iterations", default=25, type=int)

# Loss weights
parser.add_argument("--evoedit_vl_weight", default=1.0, type=float)
parser.add_argument("--evoedit_slot_weight", default=0.50, type=float)
parser.add_argument("--evoedit_edit_align_weight", default=0.20, type=float)
parser.add_argument("--evoedit_transport_weight", default=0.05, type=float)
parser.add_argument("--evoedit_mass_weight", default=0.05, type=float)
parser.add_argument("--evoedit_diversity_weight", default=0.01, type=float)
parser.add_argument("--evoedit_inversion_weight", default=0.10, type=float)
parser.add_argument("--evoedit_reverse_weight", default=0.20, type=float)

# Saved model
parser.add_argument("--savedmodel_path", type=str, default="./save/$dataset/$version")
parser.add_argument("--ckpt_file", type=str, default=None)
parser.add_argument("--delta_file", type=str, default=None)
parser.add_argument("--weights", type=list, default=[0.5, 0.5])
parser.add_argument("--device", default="cuda")
parser.add_argument("--scorer_types", type=list, default=["Bleu_4", "ROUGE_L"])

# Optimization
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--gradient_clip_val", default=None, type=int)

# Decoding
parser.add_argument("--beam_size", type=int, default=3)
parser.add_argument("--do_sample", type=str2bool, default=False)
parser.add_argument("--no_repeat_ngram_size", type=int, default=2)
parser.add_argument("--num_beam_groups", type=int, default=1)
parser.add_argument("--min_new_tokens", type=int, default=40)
parser.add_argument("--max_new_tokens", type=int, default=100)
parser.add_argument("--max_length", type=int, default=60)
parser.add_argument("--repetition_penalty", type=float, default=2.0)
parser.add_argument("--length_penalty", type=float, default=2.0)
parser.add_argument("--diversity_penalty", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=0.0)

# PyTorch Lightning
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument(
    "--accelerator",
    type=str,
    default="gpu",
    choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"],
)
parser.add_argument("--strategy", type=str, default="ddp")
parser.add_argument("--precision", type=str, default="bf16-mixed")
parser.add_argument("--limit_val_batches", type=float, default=1.0)
parser.add_argument("--limit_test_batches", type=float, default=1.0)
parser.add_argument("--limit_train_batches", type=float, default=1.0)
parser.add_argument("--max_epochs", type=int, default=15)
parser.add_argument("--every_n_train_steps", type=int, default=0)
parser.add_argument("--val_check_interval", type=float, default=1.0)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--num_sanity_val_steps", type=int, default=0)
