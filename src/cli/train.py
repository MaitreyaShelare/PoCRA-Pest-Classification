"""
Training entry point with DDP support.
Handles all phases: Phase 1 (ArcFace), Phase 2 (Router), Phase 3 (Crop Heads).
"""

import argparse
from pathlib import Path
from typing import Optional
import torch
from omegaconf import DictConfig, OmegaConf
import os

from src.core.config import load_config, save_config
from src.core.reproducibility import set_seed
from src.core.experiment import create_experiment_dir, save_experiment_metadata
from src.data.dataset import PestDataset
from src.data.transforms import get_transforms
from src.data.datamodule import get_dataloader
from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.heads.arcface_head import ArcFaceHead
from src.models.heads.router_head import RouterHead
from src.train.phase1 import Phase1Trainer
from src.train.phase2 import Phase2Trainer
from src.train.phase3 import Phase3Trainer
from src.distributed.setup import setup_ddp, cleanup_ddp


def get_ddp_config() -> tuple:
    """Get DDP rank and world size from environment."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description="Train pest classification model")
    parser.add_argument("--config-dir", type=Path, default="configs",
                        help="Path to configs directory")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1,
                        help="Training phase: 1=ArcFace, 2=Router, 3=Crop heads")
    parser.add_argument("--overrides", nargs="+", default=[],
                        help="Config overrides (e.g., learning_rate=1e-4)")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume training from checkpoint")
    args = parser.parse_args()

    # DDP setup
    rank, world_size, local_rank = get_ddp_config()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    # Load config
    cfg = load_config(args.config_dir, args.overrides)
    set_seed(cfg.seed)

    # Create experiment directory (rank 0 only)
    exp_dir = None
    if rank == 0:
        exp_dir = create_experiment_dir(Path(cfg.output_dir), cfg.experiment_name)
        save_config(cfg, exp_dir)
        save_experiment_metadata(exp_dir)
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_dir.name}")
        print(f"Phase: {args.phase}")
        print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
        print(f"{'='*60}\n")

    # Data
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

    # Sampler (for distributed training)
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.seed,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model
    if args.phase == 1:
        backbone = DINOv2Backbone(
            model_name=cfg.model.name,
            embedding_dim=cfg.model.embedding_dim,
        ).to(device)
        
        head = ArcFaceHead(
            in_features=cfg.model.embedding_dim,
            num_classes=cfg.num_pests,
            margin=cfg.model.arcface.margin,
            scale=cfg.model.arcface.scale,
        ).to(device)
        
        model = torch.nn.Sequential(backbone, head)
        trainer_class = Phase1Trainer

    elif args.phase == 2:
        backbone = DINOv2Backbone(
            model_name=cfg.model.name,
            embedding_dim=cfg.model.embedding_dim,
        ).to(device)
        backbone.freeze()  # Frozen from Phase 1
        
        head = RouterHead(
            in_features=cfg.model.embedding_dim,
            num_types=len(cfg.data.image_types),
        ).to(device)
        
        model = torch.nn.Sequential(backbone, head)
        trainer_class = Phase2Trainer

    elif args.phase == 3:
        backbone = DINOv2Backbone(
            model_name=cfg.model.name,
            embedding_dim=cfg.model.embedding_dim,
        ).to(device)
        backbone.freeze()
        
        # Per-crop heads — dict of linear layers
        crop_heads = {
            crop: torch.nn.Linear(cfg.model.embedding_dim, cfg.num_pests)
            for crop in cfg.data.crops
        }
        for head in crop_heads.values():
            head.to(device)
        
        model = (backbone, crop_heads)
        trainer_class = Phase3Trainer

    # DDP wrap (if needed)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    # Trainer
    trainer = trainer_class(
        cfg=cfg,
        model=model,
        device=device,
        output_dir=exp_dir if rank == 0 else None,
        rank=rank,
        world_size=world_size,
    )

    # Resume checkpoint if provided
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        if rank == 0:
            print(f"Resumed from {args.resume_from}")

    # Train
    if rank == 0:
        print(f"\nStarting Phase {args.phase} training...")
    
    for epoch in range(trainer.start_epoch, cfg.train[f"phase{args.phase}"].max_epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        if rank == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_f1={val_metrics.get('f1', 0):.4f} | "
                f"val_acc={val_metrics.get('accuracy', 0):.4f}"
            )

            # Early stopping
            if trainer.should_stop():
                print(f"Early stopping at epoch {epoch}")
                break

    if world_size > 1:
        cleanup_ddp()

    if rank == 0:
        print(f"\nTraining complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()