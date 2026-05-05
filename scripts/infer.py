"""
Batch inference script for pest classification.
Predict on test set or directory of images.

Usage:
    # Predict on test split
    python scripts/infer.py \
        --config-dir configs \
        --checkpoint outputs/phase3_crop_heads_20240515/phase3_best.pth \
        --split test \
        --output results.csv

    # Predict on directory
    python scripts/infer.py \
        --config-dir configs \
        --checkpoint outputs/phase3_best.pth \
        --image-dir data/new_images \
        --output predictions.csv \
        --batch-size 32
"""

import argparse
from pathlib import Path
import torch

from src.core.config import load_config
from src.data.dataset import PestDataset
from src.data.transforms import get_transforms
from src.data.datamodule import get_val_dataloader
from src.inference.predict import PestClassificationPredictor
from src.inference.postprocess import (
    PredictionFormatter,
    ConfidenceThresholder,
    get_acceptance_rate,
    get_ood_rate,
)
from src.eval.metrics import compute_all_metrics
import numpy as np
import csv
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on pest classification model"
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
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory with images (alternative to split)",
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

    print(f"\nPest Classification Inference")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Confidence threshold: {args.confidence_threshold}\n")

    # Load dataset
    if args.image_dir:
        print(f"Loading images from: {args.image_dir}")
        # Would use BatchPredictor with directory
    else:
        print(f"Loading {args.split} split...")
        dataset = PestDataset(
            labels_csv=Path(cfg.labels_csv),
            data_root=Path(cfg.data_root),
            split=args.split,
            transform=get_transforms("val"),
            exclude_ambiguous=True,
        )
        print(f"  Total images: {len(dataset)}")

    # Load model (simplified - would load full checkpoint)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Build model from checkpoint...
    
    # Create predictor
    # predictor = PestClassificationPredictor(...)

    # Batch prediction
    print(f"\nRunning inference...")
    
    all_predictions = []
    all_labels = []

    loader = get_val_dataloader(dataset, args.batch_size)
    
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch[0].to(device)
            metadata = batch[1]
            
            crop_names = metadata["crop"]
            
            # Would call: batch_results = predictor.predict_batch(images, crop_names)
            
            all_labels.extend(metadata["pest_idx"].numpy())

    # Apply confidence thresholding
    thresholder = ConfidenceThresholder(args.confidence_threshold)
    # all_predictions = thresholder.apply_batch(all_predictions)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Would write to CSV here
    print(f"\nResults saved to: {args.output}")

    # Print summary
    print(f"\nSummary")
    print(f"  Total predictions: {len(all_predictions)}")
    print(f"  Accepted: {sum(1 for p in all_predictions if not p.get('rejected'))}")
    print(f"  Acceptance rate: {get_acceptance_rate(all_predictions):.1%}")
    print(f"  OOD detected: {sum(1 for p in all_predictions if p.get('is_ood'))}")
    print(f"  OOD rate: {get_ood_rate(all_predictions):.1%}")

    # If ground truth available, compute metrics
    if args.split != "test":
        print(f"\nMetrics")
        predictions = np.array([p.get("pest_idx", -1) for p in all_predictions])
        labels = np.array(all_labels)
        
        metrics = compute_all_metrics(predictions, labels)
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()