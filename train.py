"""Training entry point for EvoEdit-OT."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
UPSTREAM_DATA_MODULE = UPSTREAM / "dataset" / "data_module.py"
if not UPSTREAM_DATA_MODULE.exists():
    raise RuntimeError(
        "Missing BiOTPrompt submodule. Run `git submodule update --init --recursive`."
    )
# Keep this repository ahead of the submodule so `configs` and `models` resolve
# to EvoEdit-OT, while upstream-only packages (dataset/evalcap/etc.) remain importable.
if str(UPSTREAM) not in sys.path:
    sys.path.append(str(UPSTREAM))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightning.pytorch as pl
from lightning.pytorch import seed_everything

from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.R2GenGPT import R2GenGPT


def train(args) -> None:
    datamodule = DataModule(args)
    callback_bundle = add_callbacks(args)

    strategy = args.strategy
    if args.devices == 1 and str(strategy).startswith("ddp"):
        strategy = "auto"

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        limit_train_batches=args.limit_train_batches,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callback_bundle["callbacks"],
        logger=callback_bundle["loggers"],
    )

    if args.ckpt_file is not None:
        model = R2GenGPT.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = R2GenGPT(args)

    if args.test:
        trainer.test(model, datamodule=datamodule)
    elif args.validate:
        trainer.validate(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule)


def main() -> None:
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)


if __name__ == "__main__":
    main()
