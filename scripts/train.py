"""
Unified training entrypoint (scripts-first).

Supports:
- Phase 1: backbone training (ArcFace + SupCon via Phase1Trainer)
- Phase 2: image-type router head training (backbone frozen)
- Phase 3: experimental placeholder (kept for compatibility; see --phase 3 notes)

Run single GPU:
  PYTHONPATH=src python scripts/train.py --phase 1

Run DDP:
  PYTHONPATH=src torchrun --nproc_per_node=4 scripts/train.py --phase 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import warnings

import torch

from src.core.config import load_config, save_config, print_config
from src.core.reproducibility import set_seed
from src.core.experiment import create_experiment_dir, save_experiment_metadata
from src.core.device import get_ddp_info, is_main_process
from src.data.dataset import PestDataset
from src.data.transforms import get_transforms
from src.data.datamodule import get_train_dataloader, get_val_dataloader
from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.heads.router_head import RouterHead
from src.train.phase1 import Phase1Trainer
from src.train.phase2 import Phase2Trainer
from src.train.phase3 import Phase3Trainer
from src.distributed.setup import setup_ddp, cleanup_ddp


warnings.filterwarnings("ignore", message="xFormers is not available")


def _build_datasets(cfg):
    train_ds = PestDataset(
        labels_csv=Path(cfg.paths.labels_csv),
        data_root=Path(cfg.paths.data_root),
        split="train",
        transform=get_transforms("train"),
        exclude_ambiguous=True,
    )
    val_ds = PestDataset(
        labels_csv=Path(cfg.paths.labels_csv),
        data_root=Path(cfg.paths.data_root),
        split="val",
        transform=get_transforms("val"),
        exclude_ambiguous=True,
    )
    return train_ds, val_ds


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train pest classification model (scripts entrypoint)")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--overrides", nargs="+", default=[])
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--phase1-checkpoint",
        type=Path,
        default=None,
        help="Required for phase 2/3. Backbone checkpoint from phase 1.",
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=Path,
        default=None,
        help="Optional for phase 3. Router checkpoint from phase 2 (if used).",
    )
    args = parser.parse_args(argv)

    rank, world_size, local_rank = get_ddp_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if world_size > 1:
        setup_ddp(rank, world_size)

    cfg = load_config(args.config_dir, args.overrides)
    set_seed(args.seed)

    exp_dir = None
    if is_main_process():
        exp_root = Path(cfg.paths.output_dir)
        exp_name = f"phase{args.phase}"
        exp_dir = create_experiment_dir(exp_root, exp_name)
        save_config(cfg, exp_dir)
        save_experiment_metadata(exp_dir, cfg)
        print_config(cfg)
        print(f"\nTraining")
        print(f"  Phase: {args.phase}")
        print(f"  Device: {device}")
        print(f"  DDP: {world_size > 1} (world_size={world_size})")
        print(f"  Output: {exp_dir}\n")

    train_ds, val_ds = _build_datasets(cfg)
    train_loader = get_train_dataloader(
        train_ds,
        batch_size=cfg.data.train_batch_size,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )
    val_loader = get_val_dataloader(
        val_ds,
        batch_size=cfg.data.val_batch_size,
        num_workers=cfg.data.num_workers,
    )

    if args.phase == 1:
        model = DINOv2Backbone(
            model_name=cfg.model.dinov2.name,
            embedding_dim=cfg.model.dinov2.embedding_dim,
            freeze_backbone=False,
        ).to(device)

        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        trainer = Phase1Trainer(
            cfg=cfg,
            model=model,
            device=device,
            output_dir=exp_dir if is_main_process() else None,
            rank=rank,
            world_size=world_size,
        )
        max_epochs = cfg.train.phase1.max_epochs
        patience = cfg.train.phase1.patience
        metric_key = "f1"

    elif args.phase == 2:
        if args.phase1_checkpoint is None:
            raise SystemExit("--phase1-checkpoint is required for phase 2")

        backbone = DINOv2Backbone(
            model_name=cfg.model.dinov2.name,
            embedding_dim=cfg.model.dinov2.embedding_dim,
            freeze_backbone=False,
        ).to(device)
        ckpt = torch.load(args.phase1_checkpoint, map_location=device)
        backbone.load_state_dict(ckpt["model_state_dict"])
        backbone.freeze()

        router = RouterHead(
            in_features=cfg.model.dinov2.embedding_dim,
            num_types=len(cfg.data.image_types),
        ).to(device)

        model = torch.nn.Sequential(backbone, router)
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        trainer = Phase2Trainer(
            cfg=cfg,
            model=model,
            device=device,
            output_dir=exp_dir if is_main_process() else None,
            rank=rank,
            world_size=world_size,
        )
        max_epochs = cfg.train.phase2.max_epochs
        patience = cfg.train.phase2.patience
        metric_key = "f1"

    else:  # phase 3
        # NOTE: Phase 3 codepaths in this repo are currently experimental/incomplete.
        # We keep an entrypoint so the workflow remains consistent, but treat it as
        # research-only unless Phase 3 trainer/model is finalized.
        if args.phase1_checkpoint is None:
            raise SystemExit("--phase1-checkpoint is required for phase 3")

        backbone = DINOv2Backbone(
            model_name=cfg.model.dinov2.name,
            embedding_dim=cfg.model.dinov2.embedding_dim,
            freeze_backbone=False,
        ).to(device)
        ckpt = torch.load(args.phase1_checkpoint, map_location=device)
        backbone.load_state_dict(ckpt["model_state_dict"])
        backbone.freeze()

        # Minimal placeholder: train a single linear head across global pest classes.
        # This keeps the repo runnable while Phase 3 per-crop heads are iterated.
        head = torch.nn.Linear(cfg.model.dinov2.embedding_dim, cfg.data.num_pests).to(device)
        model = torch.nn.Sequential(backbone, head)

        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        trainer = Phase3Trainer(
            cfg=cfg,
            model=model,
            device=device,
            output_dir=exp_dir if is_main_process() else None,
            rank=rank,
            world_size=world_size,
        )
        max_epochs = cfg.train.phase3.max_epochs
        patience = cfg.train.phase3.patience
        metric_key = "f1"

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    try:
        for epoch in range(trainer.start_epoch, max_epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)

            if is_main_process():
                loss_val = val_metrics.get("loss", float("nan"))
                metric_val = val_metrics.get(metric_key, 0.0)
                print(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_metrics.get('loss', 0.0):.4f} | "
                    f"val_loss={loss_val:.4f} | "
                    f"val_{metric_key}={metric_val:.4f}"
                )

                trainer.log_metrics(
                    {f"train/{k}": v for k, v in train_metrics.items()}
                    | {f"val/{k}": v for k, v in val_metrics.items()},
                    epoch=epoch,
                )

                trainer.save_checkpoint(
                    Path(exp_dir) / f"checkpoint_epoch_{epoch}.pth",
                    epoch=epoch,
                    metrics=val_metrics,
                )

            if trainer.should_stop(val_metrics.get(metric_key, 0.0), patience):
                if is_main_process():
                    print(f"Early stopping at epoch {epoch}")
                break
    finally:
        trainer.close()
        if world_size > 1:
            cleanup_ddp()


if __name__ == "__main__":
    main()

