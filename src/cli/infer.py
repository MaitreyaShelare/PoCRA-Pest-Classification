"""
Inference entry point for predictions on test set.
Computes metrics and generates predictions CSV.
"""

import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
import csv
from tqdm import tqdm

from src.core.config import load_config
from src.data.dataset import PestDataset
from src.data.transforms import get_transforms
from src.data.datamodule import get_dataloader
from src.inference.predict import predict_batch
from src.eval.metrics import macro_f1, per_crop_accuracy


def main():
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("--config-dir", type=Path, default="configs",
                        help="Path to configs directory")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=3,
                        help="Model phase (for correct architecture)")
    parser.add_argument("--output-dir", type=Path, default="outputs",
                        help="Output directory for predictions CSV")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Dataset split to evaluate")
    args = parser.parse_args()

    # Setup
    cfg = load_config(args.config_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = PestDataset(
        labels_csv=Path(cfg.labels_csv),
        data_root=Path(cfg.data_root),
        split=args.split,
        transform=get_transforms("val"),
        exclude_ambiguous=True,
    )
    loader = get_dataloader(
        dataset,
        batch_size=cfg.val_batch_size,
        split="val",
        num_workers=cfg.num_workers,
    )

    print(f"\nInference on {args.split} split ({len(dataset)} images)")
    print(f"Checkpoint: {args.checkpoint}\n")

    # Inference
    all_predictions = []
    all_labels = []
    all_crops = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencing"):
            images = batch["image"].to(device)
            preds = predict_batch(images, args.checkpoint, args.phase, device)
            
            all_predictions.extend(preds)
            all_labels.extend(batch["pest_idx"].tolist())
            all_crops.extend(batch["crop"])

    # Metrics
    f1 = macro_f1(all_predictions, all_labels)
    per_crop_acc = per_crop_accuracy(all_predictions, all_labels, all_crops)

    print(f"\nResults:")
    print(f"  Macro-F1: {f1:.4f}")
    for crop, acc in per_crop_acc.items():
        print(f"  {crop}: {acc:.4f}")

    # Save predictions
    output_csv = args.output_dir / f"predictions_{args.split}.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "pred_class", "true_class"])
        writer.writeheader()
        for pred, label, crop in zip(all_predictions, all_labels, all_crops):
            writer.writerow({
                "filepath": f"{crop}/{label}",
                "pred_class": pred,
                "true_class": label,
            })

    print(f"\nPredictions saved to {output_csv}")


if __name__ == "__main__":
    main()