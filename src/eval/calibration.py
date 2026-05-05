"""
OOD confidence threshold calibration.
Sweep thresholds to find optimal operating point.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from pathlib import Path
import json


def calibrate_ood_threshold(
    in_distribution_confidences: np.ndarray,
    out_of_distribution_confidences: np.ndarray,
    target_tpr: float = 0.95,
    target_fpr: float = 0.05,
) -> Tuple[float, Dict]:
    """
    Find optimal OOD confidence threshold.
    
    Args:
        in_distribution_confidences: Confidence scores for ID samples (N_id,)
        out_of_distribution_confidences: Confidence scores for OOD samples (N_ood,)
        target_tpr: Target true positive rate (reject OOD)
        target_fpr: Target false positive rate (reject ID)
    
    Returns:
        (optimal_threshold, metrics_dict)
    """
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_score = float("-inf")
    
    results = {
        "thresholds": [],
        "tpr": [],  # % of OOD correctly rejected
        "fpr": [],  # % of ID incorrectly rejected
        "f1": [],
    }
    
    for threshold in thresholds:
        # ID: reject if confidence < threshold (want to keep, so FP is rejection)
        id_accepted = (in_distribution_confidences >= threshold).sum()
        id_rejected = (in_distribution_confidences < threshold).sum()
        fpr = id_rejected / len(in_distribution_confidences) if len(in_distribution_confidences) > 0 else 0
        
        # OOD: reject if confidence < threshold (want to reject, so TP is rejection)
        ood_rejected = (out_of_distribution_confidences < threshold).sum()
        tpr = ood_rejected / len(out_of_distribution_confidences) if len(out_of_distribution_confidences) > 0 else 0
        
        # F1 balancing TPR and (1-FPR)
        f1 = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-10)
        
        results["thresholds"].append(float(threshold))
        results["tpr"].append(float(tpr))
        results["fpr"].append(float(fpr))
        results["f1"].append(float(f1))
        
        # Find threshold closest to target TPR/FPR
        if abs(tpr - target_tpr) < 0.05 and abs(fpr - target_fpr) < 0.05:
            if f1 > best_score:
                best_score = f1
                best_threshold = threshold
    
    return best_threshold, results


def calibrate_class_confidence_threshold(
    confidences: np.ndarray,
    labels: np.ndarray,
    target_accuracy: float = 0.95,
) -> Tuple[float, Dict]:
    """
    Find confidence threshold that achieves target accuracy.
    
    Args:
        confidences: Confidence scores (N,)
        labels: Correctness labels: 1=correct, 0=incorrect (N,)
        target_accuracy: Target accuracy on accepted samples
    
    Returns:
        (optimal_threshold, metrics_dict)
    """
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_diff = float("inf")
    
    results = {
        "thresholds": [],
        "accuracy_on_accepted": [],
        "rejection_rate": [],
    }
    
    for threshold in thresholds:
        accepted_mask = confidences >= threshold
        
        if accepted_mask.sum() == 0:
            continue
        
        # Accuracy on accepted samples
        accepted_correct = labels[accepted_mask].sum()
        accuracy_on_accepted = accepted_correct / accepted_mask.sum()
        
        # Rejection rate
        rejection_rate = (~accepted_mask).sum() / len(accepted_mask)
        
        results["thresholds"].append(float(threshold))
        results["accuracy_on_accepted"].append(float(accuracy_on_accepted))
        results["rejection_rate"].append(float(rejection_rate))
        
        # Find threshold closest to target
        diff = abs(accuracy_on_accepted - target_accuracy)
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
    
    return best_threshold, results


def sweep_prototype_distance_threshold(
    distances: np.ndarray,
    labels: np.ndarray,
    target_recall: float = 0.95,
) -> Tuple[float, Dict]:
    """
    Calibrate prototype distance threshold for OOD detection.
    Lower distance = more confident (known pest).
    
    Args:
        distances: Distances to nearest prototype (N,)
        labels: Ground truth labels (N,) - high value = OOD
        target_recall: Target recall on OOD detection
    
    Returns:
        (optimal_threshold, metrics_dict)
    """
    thresholds = np.linspace(distances.min(), distances.max(), 101)
    best_threshold = distances.mean()
    best_f1 = 0.0
    
    results = {
        "thresholds": [],
        "ood_recall": [],  # % of true OOD samples rejected
        "id_specificity": [],  # % of true ID samples accepted
    }
    
    for threshold in thresholds:
        # Predict OOD if distance > threshold
        ood_pred = distances > threshold
        
        # Metrics
        ood_tp = ((ood_pred == 1) & (labels == 1)).sum()
        ood_fn = ((ood_pred == 0) & (labels == 1)).sum()
        id_tn = ((ood_pred == 0) & (labels == 0)).sum()
        id_fp = ((ood_pred == 1) & (labels == 0)).sum()
        
        ood_recall = ood_tp / (ood_tp + ood_fn) if (ood_tp + ood_fn) > 0 else 0
        id_specificity = id_tn / (id_tn + id_fp) if (id_tn + id_fp) > 0 else 0
        
        f1 = 2 * (ood_recall * id_specificity) / (ood_recall + id_specificity + 1e-10)
        
        results["thresholds"].append(float(threshold))
        results["ood_recall"].append(float(ood_recall))
        results["id_specificity"].append(float(id_specificity))
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, results


def save_calibration_results(
    results: Dict,
    output_path: Path,
) -> None:
    """
    Save calibration results to JSON.
    
    Args:
        results: Calibration results dict
        output_path: Path to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)