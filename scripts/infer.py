"""
Batch inference script (scripts-first).

Currently supported (reliable) inference mode:
- **router**: predicts `image_type` (pest_body / symptom / healthy) from a Phase 2 checkpoint.

Usage:
    # Predict image_type on a split (Phase 2 router checkpoint)
    PYTHONPATH=src python scripts/infer.py \
        --mode router \
        --config-dir configs \
        --checkpoint outputs/phase2_*/checkpoint_epoch_*.pth \
        --split test \
        --output outputs/router_predictions.csv
"""

import argparse
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.core.config import load_config
from src.data.dataset import PestDataset
from src.data.transforms import get_transforms
from src.data.datamodule import get_val_dataloader
from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.heads.router_head import RouterHead


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on pest classification model"
    )
    parser.add_argument(
        "--mode",
        choices=["router"],
        default="router",
        help="Inference mode. `router` loads a Phase 2 checkpoint and predicts image_type.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default="configs",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (for mode=router: Phase 2 checkpoint)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="predictions.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = parser.parse_args()

    cfg = load_config(args.config_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\nInference")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"\nLoading {args.split} split...")

    # Load dataset
    dataset = PestDataset(
        labels_csv=Path(cfg.paths.labels_csv),
        data_root=Path(cfg.paths.data_root),
        split=args.split,
        transform=get_transforms("val"),
        exclude_ambiguous=True,
    )
    print(f"  Total images: {len(dataset)}")

    loader = get_val_dataloader(dataset, args.batch_size, num_workers=cfg.data.num_workers)

    # Load model for requested mode
    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.mode == "router":
        backbone = DINOv2Backbone(
            model_name=cfg.model.dinov2.name,
            embedding_dim=cfg.model.dinov2.embedding_dim,
            freeze_backbone=True,
        ).to(device)
        router = RouterHead(
            in_features=cfg.model.dinov2.embedding_dim,
            num_types=len(cfg.data.image_types),
        ).to(device)
        model = torch.nn.Sequential(backbone, router)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        idx_to_type = {i: t for i, t in enumerate(cfg.data.image_types)}

        rows = []
        with torch.no_grad():
            for images, meta in tqdm(loader, desc="Infer (router)"):
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1)
                conf = probs.max(dim=1).values

                for fp, crop, pi, c in zip(
                    meta["filepath"],
                    meta["crop"],
                    pred_idx.detach().cpu().tolist(),
                    conf.detach().cpu().tolist(),
                ):
                    rows.append(
                        {
                            "filepath": fp,
                            "crop": crop,
                            "image_type_pred": idx_to_type.get(pi, "unknown"),
                            "confidence": float(c),
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filepath", "crop", "image_type_pred", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()