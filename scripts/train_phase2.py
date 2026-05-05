"""
Phase 2 Training: Image type router (pest_body / symptom / healthy).
Backbone frozen from Phase 1.

Usage:
    torchrun --nproc_per_node=4 scripts/train_phase2.py --config-dir configs \
        --phase1-checkpoint outputs/phase1_arcface_20240515/phase1_best.pth
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
from src.models.heads.router_head import RouterHead
from src.train.phase2 import Phase2Trainer
from src.distributed.setup import setup_ddp, cleanup_ddp


def main():
    parser = argparse.ArgumentParser(
        description="Train Phase 2: Image type router"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default="configs",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=Path,
        required=True,
        help="Path to Phase 1 checkpoint (for frozen backbone)",
    )
    parser.add_argument(
        "--overrides",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    # DDP setup
    rank, world_size, local_rank = get_ddp_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if world_size > 1:
        setup_ddp(rank, world_size)

    # Load config
    cfg = load_config(args.config_dir, args.overrides)
    set_seed(cfg.seed)

    # Create experiment directory
    exp_dir = None
    if is_main_process():
        exp_dir = create_experiment_dir(Path(cfg.output_dir), "phase2_router")
        save_config(cfg, exp_dir)
        save_experiment_metadata(exp_dir)
        print_config(cfg)
        print(f"\nPhase 2 Training: Image Type Router")
        print(f"  Device: {device}")
        print(f"  DDP: {world_size > 1}\n")

    # Load data
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

    train_loader = get_train_dataloader(train_dataset, cfg.train_batch_size)
    val_loader = get_val_dataloader(val_dataset, cfg.val_batch_size)

    # Build model
    if is_main_process():
        print("Building model...")
    
    backbone = DINOv2Backbone(
        model_name=cfg.model.name,
        embedding_dim=cfg.model.embedding_dim,
    ).to(device)
    
    # Load Phase 1 checkpoint
    ckpt = torch.load(args.phase1_checkpoint, map_location=device)
    if isinstance(backbone, torch.nn.parallel.DistributedDataParallel):
        backbone.module.load_state_dict(ckpt["model_state_dict"])
    else:
        backbone.load_state_dict(ckpt["model_state_dict"])
    
    backbone.freeze()
    
    router_head = RouterHead(
        in_features=cfg.model.embedding_dim,
        num_types=len(cfg.data.image_types),
    ).to(device)

    # DDP wrap router head
    if world_size > 1:
        router_head = torch.nn.parallel.DistributedDataParallel(
            router_head,
            device_ids=[local_rank],
        )

    # Create trainer
    trainer = Phase2Trainer(
        cfg=cfg,
        model=torch.nn.Sequential(backbone, router_head),
        device=device,
        output_dir=exp_dir if is_main_process() else None,
        rank=rank,
        world_size=world_size,
    )

    # Training loop
    if is_main_process():
        print(f"Starting Phase 2 training (router)...")
        print(f"  Backbone frozen: yes")
        print(f"  Epochs: {cfg.train.phase2.max_epochs}\n")

    try:
        for epoch in range(trainer.start_epoch, cfg.train.phase2.max_epochs):
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)

            if is_main_process():
                print(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_f1={val_metrics['f1']:.4f}"
                )
                
                trainer.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["loss"],
                        "val_f1": val_metrics["f1"],
                    },
                    epoch=epoch,
                )
                
                trainer.save_checkpoint(exp_dir / "phase2_best.pth", epoch)

            if trainer.should_stop(val_metrics["f1"], cfg.train.phase2.patience):
                if is_main_process():
                    print(f"Early stopping at epoch {epoch}")
                break

    finally:
        trainer.close()
        if world_size > 1:
            cleanup_ddp()

    if is_main_process():
        print(f"Phase 2 training complete!")
        print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()