"""
Evaluation metrics for pest classification.
Handles imbalanced multi-class classification and per-crop evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def macro_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Macro-averaged F1 score (average F1 across all classes).
    Best metric for imbalanced classification.
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
    
    Returns:
        Macro F1 score (0 to 1)
    """
    return f1_score(labels, predictions, average="macro", zero_division=0)


def weighted_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Weighted F1 score (weighted by support).
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
    
    Returns:
        Weighted F1 score
    """
    return f1_score(labels, predictions, average="weighted", zero_division=0)


def accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Overall accuracy.
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
    
    Returns:
        Accuracy (0 to 1)
    """
    return accuracy_score(labels, predictions)


def per_class_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, float]:
    """
    F1 score per class.
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
    
    Returns:
        Dict mapping class_idx -> F1 score
    """
    f1_scores = f1_score(labels, predictions, average=None, zero_division=0)
    return {i: score for i, score in enumerate(f1_scores)}


def per_class_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, float]:
    """
    Accuracy per class (recall).
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
    
    Returns:
        Dict mapping class_idx -> accuracy (recall)
    """
    recalls = recall_score(labels, predictions, average=None, zero_division=0)
    return {i: score for i, score in enumerate(recalls)}


def per_crop_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    crop_indices: np.ndarray,
    crop_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Accuracy per crop.
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
        crop_indices: Crop index for each sample (N,)
        crop_names: Optional crop names (default: crop_0, crop_1, ...)
    
    Returns:
        Dict mapping crop_name -> accuracy
    """
    unique_crops = np.unique(crop_indices)
    results = {}
    
    for crop_idx in unique_crops:
        mask = crop_indices == crop_idx
        crop_acc = accuracy_score(labels[mask], predictions[mask])
        
        crop_name = crop_names[crop_idx] if crop_names else f"crop_{crop_idx}"
        results[crop_name] = crop_acc
    
    return results


def per_crop_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    crop_indices: np.ndarray,
    crop_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Macro F1 per crop.
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
        crop_indices: Crop index for each sample (N,)
        crop_names: Optional crop names
    
    Returns:
        Dict mapping crop_name -> F1 score
    """
    unique_crops = np.unique(crop_indices)
    results = {}
    
    for crop_idx in unique_crops:
        mask = crop_indices == crop_idx
        crop_f1 = f1_score(labels[mask], predictions[mask], average="macro", zero_division=0)
        
        crop_name = crop_names[crop_idx] if crop_names else f"crop_{crop_idx}"
        results[crop_name] = crop_f1
    
    return results


def ood_auroc(
    in_distribution_scores: np.ndarray,
    out_of_distribution_scores: np.ndarray,
) -> float:
    """
    AUROC for OOD detection.
    Higher score = better OOD detection.
    
    Args:
        in_distribution_scores: Confidence scores for in-distribution samples (N_in,)
        out_of_distribution_scores: Confidence scores for OOD samples (N_out,)
    
    Returns:
        AUROC score (0 to 1)
    """
    all_scores = np.concatenate([in_distribution_scores, out_of_distribution_scores])
    all_labels = np.concatenate([
        np.ones_like(in_distribution_scores),
        np.zeros_like(out_of_distribution_scores),
    ])
    
    return roc_auc_score(all_labels, all_scores)


def ood_tpr_at_fpr(
    in_distribution_scores: np.ndarray,
    out_of_distribution_scores: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """
    TPR at fixed FPR for OOD detection.
    
    Args:
        in_distribution_scores: Confidence scores for in-distribution samples
        out_of_distribution_scores: Confidence scores for OOD samples
        target_fpr: Target false positive rate (e.g., 0.05 for 5%)
    
    Returns:
        True positive rate at target FPR
    """
    # Threshold: set score such that FPR = target_fpr
    threshold = np.percentile(in_distribution_scores, 100 * target_fpr)
    
    # TPR: fraction of OOD samples below threshold
    tpr = (out_of_distribution_scores < threshold).mean()
    
    return tpr


class MetricTracker:
    """
    Tracks multiple metrics across batches and epochs.
    Useful for logging during training.
    """
    
    def __init__(self, metric_names: List[str]) -> None:
        """
        Initialize metric tracker.
        
        Args:
            metric_names: List of metric names to track
        """
        self.metric_names = metric_names
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {name: [] for name in self.metric_names}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update metrics with new values.
        
        Args:
            metrics: Dict of metric_name -> value
        """
        for name, value in metrics.items():
            if name in self.metrics:
                self.metrics[name].append(value)
    
    def get_avg(self) -> Dict[str, float]:
        """Get average of all metrics."""
        return {
            name: np.mean(values) if values else 0.0
            for name, values in self.metrics.items()
        }
    
    def get_summary(self) -> str:
        """Get formatted summary string."""
        avg = self.get_avg()
        lines = []
        for name, value in avg.items():
            lines.append(f"{name}: {value:.4f}")
        return " | ".join(lines)


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    crop_indices: Optional[np.ndarray] = None,
    crop_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute all metrics at once.
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
        crop_indices: Optional crop indices (N,)
        crop_names: Optional crop names
    
    Returns:
        Dict of all computed metrics
    """
    results = {
        "accuracy": accuracy(predictions, labels),
        "macro_f1": macro_f1(predictions, labels),
        "weighted_f1": weighted_f1(predictions, labels),
    }
    
    if crop_indices is not None:
        results["per_crop_accuracy"] = per_crop_accuracy(
            predictions, labels, crop_indices, crop_names
        )
        results["per_crop_f1"] = per_crop_f1(
            predictions, labels, crop_indices, crop_names
        )
    
    return results