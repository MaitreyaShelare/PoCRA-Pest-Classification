"""
Evaluation script: metrics, confusion matrix, OOD calibration.
"""

import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from src.core.config import load_config
from src.data.dataset import PestDataset
from src.data.transforms import get_transforms
from src.data.datamodule import get_dataloader
from src.inference.predict import predict_batch
from src.eval.metrics import macro_f1, per_crop_accuracy, ood_auroc
from src.eval.confusion import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate model comprehensively")
    parser.add_argument("--config-dir", type=Path, default="configs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--output-dir", type=Path, default="outputs/eval")
    args = parser.parse_args()

    cfg = load_config(args.config_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Comprehensive Evaluation")
    print(f"{'='*60}\n")

    # Evaluate on val and test
    for split in ["val", "test"]:
        dataset = PestDataset(
            labels_csv=Path(cfg.labels_csv),
            data_root=Path(cfg.data_root),
            split=split,
            transform=get_transforms("val"),
        )
        loader = get_dataloader(dataset, batch_size=cfg.val_batch_size, split="val")

        all_preds = []
        all_labels = []
        all_confs = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                preds, confs = predict_batch(images, args.checkpoint, args.phase, device)
                all_preds.extend(preds)
                all_labels.extend(batch["pest_idx"].tolist())
                all_confs.extend(confs)

        # Metrics
        f1 = macro_f1(all_preds, all_labels)
        report = classification_report(all_labels, all_preds, output_dict=True)

        print(f"\n{split.upper()} SET")
        print(f"  Macro-F1: {f1:.4f}")
        print(f"  Accuracy: {report['accuracy']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, args.output_dir / f"confusion_{split}.png")

        # Save report
        with open(args.output_dir / f"report_{split}.txt", "w") as f:
            f.write(classification_report(all_labels, all_preds))

    print(f"\nEvaluation complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()