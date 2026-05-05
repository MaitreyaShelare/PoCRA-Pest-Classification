"""
Phase 3 Training: Per-crop pest and symptom heads.
Backbone and router frozen from Phases 1-2.

Usage:
    torchrun --nproc_per_node=4 scripts/train_phase3.py --config-dir configs \
        --phase1-checkpoint outputs/phase1_best.pth \
        --phase2-checkpoint outputs/phase2_best.pth
"""

import argparse
from pathlib import Path
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
from src.models.heads.crop_heads import CropHeadsRegistry
from src.train.phase3 import Phase3Trainer
from src.distributed.setup import setup_ddp, cleanup_ddp


def main():
    parser = argparse.ArgumentParser(
        description="Train Phase 3: Per-crop pest and symptom heads"
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
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overrides",
        nargs="+",
        default=[],
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
        exp_dir = create_experiment_dir(Path(cfg.output_dir), "phase3_crop_heads")
        save_config(cfg, exp_dir)
        save_experiment_metadata(exp_dir)
        print(f"\nPhase 3 Training: Per-crop Heads")
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
    backbone = DINOv2Backbone(cfg.model.name, cfg.model.embedding_dim).to(device)
    backbone.load_state_dict(torch.load(args.phase1_checkpoint, map_location=device)["model_state_dict"])
    backbone.freeze()

    router_head = RouterHead(cfg.model.embedding_dim, len(cfg.data.image_types)).to(device)
    # Load phase2 checkpoint for router if needed

    # Create crop heads registry
    # Would compute num_pests_per_crop from dataset
    num_pests_per_crop = {crop: 10 for crop in cfg.data.crops}  # Simplified
    num_symptoms_per_crop = {crop: 5 for crop in cfg.data.crops}

    crop_heads = CropHeadsRegistry(
        crops=cfg.data.crops,
        num_pests_per_crop=num_pests_per_crop,
        num_symptoms_per_crop=num_symptoms_per_crop,
        in_features=cfg.model.embedding_dim,
    ).to(device)

    # DDP wrap heads only
    if world_size > 1:
        crop_heads = torch.nn.parallel.DistributedDataParallel(crop_heads, device_ids=[local_rank])

    # Create trainer
    trainer = Phase3Trainer(
        cfg=cfg,
        model=crop_heads,
        device=device,
        output_dir=exp_dir if is_main_process() else None,
        rank=rank,
        world_size=world_size,
    )

    # Training loop
    if is_main_process():
        print(f"Starting Phase 3 training (crop heads)...")
        print(f"  Crops: {len(cfg.data.crops)}")
        print(f"  Epochs: {cfg.train.phase3.max_epochs}\n")

    try:
        for epoch in range(trainer.start_epoch, cfg.train.phase3.max_epochs):
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)

            if is_main_process():
                print(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_f1={val_metrics['f1']:.4f}"
                )
                
                trainer.save_checkpoint(exp_dir / "phase3_best.pth", epoch)

            if trainer.should_stop(val_metrics["f1"], cfg.train.phase3.patience):
                if is_main_process():
                    print(f"Early stopping at epoch {epoch}")
                break

    finally:
        trainer.close()
        if world_size > 1:
            cleanup_ddp()

    if is_main_process():
        print(f"Phase 3 training complete!")
        print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()