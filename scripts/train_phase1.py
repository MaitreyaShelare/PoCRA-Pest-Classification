"""
Phase 1 Training: ArcFace + SupCon for learning species-level embeddings.
Entry point with DDP support for multi-GPU training.

Usage:
    # Single GPU
    python scripts/train_phase1.py --config-dir configs

    # 4 GPUs with DDP
    torchrun --nproc_per_node=4 scripts/train_phase1.py --config-dir configs

    # Resume from checkpoint
    python scripts/train_phase1.py --config-dir configs --resume-from outputs/exp_20240515/phase1_best.pth
"""

import argparse
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf

from src.core.config import load_config, save_config, print_config
from src.core.reproducibility import set_seed
from src.core.experiment import create_experiment_dir, save_experiment_metadata
from src.core.device import get_ddp_info, is_main_process
from src.data.dataset import PestDataset
from src.data.transforms import get_transforms
from src.data.datamodule import get_train_dataloader, get_val_dataloader
from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.losses.arcface_loss import ArcFaceLoss
from src.train.phase1 import Phase1Trainer
from src.distributed.setup import setup_ddp, cleanup_ddp
from src.distributed.utils import is_main_process
from src.eval.metrics import MetricTracker


def main():
    parser = argparse.ArgumentParser(
        description="Train Phase 1: ArcFace + SupCon for pest embeddings"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default="configs",
        help="Path to configs directory",
    )
    parser.add_argument(
        "--overrides",
        nargs="+",
        default=[],
        help="Config overrides (e.g., learning_rate=1e-4)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # DDP setup
    rank, world_size, local_rank = get_ddp_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    # Load config
    cfg = load_config(args.config_dir, args.overrides)
    set_seed(args.seed)
    
    # Create experiment directory (rank 0 only)
    exp_dir = None
    if is_main_process():
        exp_dir = create_experiment_dir(Path(cfg.output_dir), "phase1_arcface")
        save_config(cfg, exp_dir)
        save_experiment_metadata(exp_dir, cfg)
        print_config(cfg)
        print(f"\nPhase 1 Training")
        print(f"  Device: {device}")
        print(f"  DDP: {world_size > 1} (world_size={world_size})")
        print(f"  Output: {exp_dir}\n")

    # Load data
    if is_main_process():
        print("Loading dataset...")
    
    train_dataset = PestDataset(
        labels_csv=Path(cfg.labels_csv),
        data_root=Path(cfg.data_root),
        split="train",
        transform=get_transforms("train"),
        exclude_ambiguous=True,
    )
    
    val_dataset = PestDataset(
        labels_csv=Path(cfg.labels_csv),
        data_root=Path(cfg.data_root),
        split="val",
        transform=get_transforms("val"),
        exclude_ambiguous=True,
    )
    
    train_loader = get_train_dataloader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers,
    )
    
    val_loader = get_val_dataloader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
    )
    
    if is_main_process():
        print(f"  Train: {len(train_dataset)} images")
        print(f"  Val: {len(val_dataset)} images")
        print(f"  Batch size: {cfg.train_batch_size}\n")

    # Build model
    if is_main_process():
        print("Building model...")
    
    backbone = DINOv2Backbone(
        model_name=cfg.model.name,
        embedding_dim=cfg.model.embedding_dim,
        freeze_backbone=False,
    ).to(device)
    
    if is_main_process():
        print(f"  Backbone: {cfg.model.name}")
        print(f"  Embedding dim: {cfg.model.embedding_dim}\n")

    # DDP wrap
    if world_size > 1:
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # Create trainer
    trainer = Phase1Trainer(
        cfg=cfg,
        model=backbone,
        device=device,
        output_dir=exp_dir if is_main_process() else None,
        rank=rank,
        world_size=world_size,
    )

    # Resume checkpoint if provided
    if args.resume_from:
        if is_main_process():
            print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Training loop
    if is_main_process():
        print(f"Starting Phase 1 training...")
        print(f"  Epochs: {cfg.train.phase1.max_epochs}")
        print(f"  Early stopping patience: {cfg.train.phase1.patience}\n")
    
    metric_tracker = MetricTracker(["loss", "f1"])
    
    try:
        for epoch in range(trainer.start_epoch, cfg.train.phase1.max_epochs):
            # Train
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            
            # Log metrics
            if is_main_process():
                print(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_f1={train_metrics.get('f1', 0):.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_f1={val_metrics['f1']:.4f}"
                )
                
                trainer.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "train_f1": train_metrics.get("f1", 0),
                        "val_loss": val_metrics["loss"],
                        "val_f1": val_metrics["f1"],
                    },
                    epoch=epoch,
                )
                
                # Save checkpoint
                trainer.save_checkpoint(
                    exp_dir / f"checkpoint_epoch_{epoch}.pth",
                    epoch=epoch,
                    metrics=val_metrics,
                )
                
                # Save best checkpoint
                if val_metrics["f1"] > trainer.best_metric:
                    trainer.best_metric = val_metrics["f1"]
                    trainer.save_checkpoint(
                        exp_dir / "phase1_best.pth",
                        epoch=epoch,
                        metrics=val_metrics,
                    )
            
            # Early stopping
            if trainer.should_stop(val_metrics["f1"], cfg.train.phase1.patience):
                if is_main_process():
                    print(f"\nEarly stopping at epoch {epoch}")
                break
    
    except KeyboardInterrupt:
        if is_main_process():
            print("\nTraining interrupted by user")
    
    finally:
        trainer.close()
        if world_size > 1:
            cleanup_ddp()

    if is_main_process():
        print(f"\nPhase 1 training complete!")
        print(f"Best checkpoint: {exp_dir / 'phase1_best.pth'}")
        print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()