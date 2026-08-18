# EvoEdit-OT

**End-to-End Birth–Death Optimal Transport for Longitudinal Radiology Report Editing**

EvoEdit-OT is a model-side extension of [BiOTPrompt](https://github.com/TengfeiLiu966/BiOTPrompt). It keeps the original longitudinal dataset, annotation files, data splits, image preprocessing, backbone models, evaluation code, and training entry points. The change is entirely in how temporal evidence is represented and how the current report is generated.

> Disease evolution is treated as a non-mass-preserving report-editing process rather than balanced patch matching followed by patch-index strings.

## What is implemented

The repository contains a runnable integration of the proposed method:

1. **Birth–Death Unbalanced OT**
   - predicts positive, sample-dependent pathology mass for historical and current patches;
   - uses generalized log-domain Sinkhorn updates with KL-relaxed marginals;
   - treats unmatched current mass as **NEW**, unmatched historical mass as **RESOLVED**, matched dissimilar mass as **MODIFY**, and matched similar mass as **KEEP**;
   - adds spatial cost without requiring region annotations.

2. **Continuous evolution tokens**
   - weighted learnable queries pool NEW/RESOLVED/MODIFY/KEEP evidence;
   - replaces non-differentiable `torch.where(...).tolist()` patch-index prompts;
   - lets the report loss back-propagate into temporal correspondence and mass prediction.

3. **Latent finding-slot encoder**
   - compresses the existing historical report into an unordered set of finding slots;
   - uses the LLM's token embeddings and trainable cross-attention queries;
   - requires no sentence, disease, bounding-box, or segmentation supervision.

4. **Transport-conditioned report editor**
   - predicts soft KEEP/DELETE/MODIFY gates for historical finding slots;
   - creates ADD slots from NEW visual evidence;
   - passes the edited slots, visual edit tokens, prior report, current image, and prior image to the LLM.

5. **Text-UOT online teacher**
   - aligns historical and current ground-truth report slots during training;
   - distills a four-way edit distribution into the visual UOT branch;
   - is removed at inference time, so there is no target-report leakage.

6. **Temporal inversion consistency**
   - swaps current and historical image order during training;
   - enforces NEW ↔ RESOLVED and KEEP ↔ KEEP consistency;
   - performs reverse slot editing back toward the historical report.

## Repository layout

```text
EvoEdit-OT/
├── upstream/                       # pinned BiOTPrompt git submodule
├── configs/config.py               # original options + EvoEdit-OT options
├── models/R2GenGPT.py              # integrated Lightning model
├── models/modules/
│   ├── birth_death_uot.py          # generalized UOT + edit tokens
│   ├── finding_slots.py            # finding encoder + report editor
│   └── text_uot.py                 # online text edit teacher
├── scripts/smoke_test_modules.py   # CPU-only differentiability check
├── tests/test_evoedit_modules.py   # module tests
└── train.py                        # default EvoEdit-OT entry point
```

## Installation

Clone with the upstream submodule:

```bash
git clone --recurse-submodules https://github.com/redwangwangwang/EvoEdit-OT.git
cd EvoEdit-OT
pip install -r requirements.txt
```

For an existing clone:

```bash
git submodule update --init --recursive
pip install -r requirements.txt
```

Prepare the same Longitudinal-MIMIC data, Swin checkpoint, LLM checkpoint, and CheXbert checkpoint expected by BiOTPrompt. No annotation conversion or split regeneration is needed.

## Module smoke test

This test does not require the medical dataset or pretrained checkpoints:

```bash
python scripts/smoke_test_modules.py
pytest
```

It checks output shapes, finite UOT plans, four-way edit normalization, temporal inversion mapping, and gradient flow through the UOT, slot encoder, and editor.

## Training and testing

Existing BiOTPrompt command-line arguments remain valid. The original shell scripts can be used as references; invoke the root `train.py` so the EvoEdit-OT model is selected:

```bash
python train.py \
  --dataset mimic-cxr \
  --annotation /path/to/longitudinal/annotations \
  --base_dir /path/to/mimic-cxr-jpg \
  --vision_model /path/to/swin-base-patch4-window7-224 \
  --llama_model /path/to/llama-or-bimedix \
  --savedmodel_path ./save/evoedit_ot
```

Testing a saved delta checkpoint:

```bash
python train.py --test --delta_file /path/to/checkpoint.pth [the same data/model arguments]
```

## Main EvoEdit-OT options

| Option | Default | Meaning |
|---|---:|---|
| `--evoedit_num_edit_queries` | 4 | Tokens pooled for each edit type |
| `--evoedit_num_finding_slots` | 8 | Historical/current report slots |
| `--evoedit_num_add_slots` | 4 | New finding slots created by the editor |
| `--evoedit_uot_epsilon` | 0.08 | Visual UOT entropy regularization |
| `--evoedit_uot_tau` | 0.8 | Visual marginal-relaxation strength |
| `--evoedit_spatial_weight` | 0.10 | Relative patch-position cost |
| `--evoedit_slot_weight` | 0.50 | Current-slot reconstruction loss |
| `--evoedit_edit_align_weight` | 0.20 | Text-to-visual edit distillation |
| `--evoedit_inversion_weight` | 0.10 | Forward/reverse edit consistency |
| `--evoedit_reverse_weight` | 0.20 | Reverse report-slot editing |

Set `--evoedit_inversion_weight 0 --evoedit_reverse_weight 0` for a lower-memory ablation.

## Training objective

The implementation optimizes

```text
L = L_LM
  + λ_vl (L_image-cls + L_text-cls + L_VL-KL)
  + λ_slot L_slot-UOT
  + λ_align L_edit-distill
  + λ_transport L_visual-UOT
  + λ_mass L_mass-reg
  + λ_div L_slot-diversity
  + λ_inv L_temporal-inversion
  + λ_rev L_reverse-edit
```

The current report is used only as the normal generation target and as an online text-side teacher during training. Inference uses the current image, historical image, and historical report already present in the original samples.

## Compatibility notes

- Old BiOTPrompt delta checkpoints can be loaded with `strict=False`; EvoEdit-OT modules will initialize from scratch.
- New EvoEdit-OT checkpoints are loaded only after all subclass modules are constructed, fixing the upstream initialization-order issue for added parameters.
- The vision encoder is evaluated once per image in `encode_img`, rather than twice as in the upstream implementation.
- Full end-to-end numerical validation requires the original protected medical data and pretrained model files. The repository includes CPU module tests, but no claim of benchmark performance is made without running that training/evaluation pipeline.

## License

BSD 3-Clause. See `LICENSE` and `NOTICE.md`.
