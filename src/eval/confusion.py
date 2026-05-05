"""
Confusion matrix computation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
        normalize: If True, normalize by row (recall)
    
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    cm = sklearn_confusion_matrix(labels, predictions)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (12, 10),
    normalize: bool = False,
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        output_path: Path to save figure
        class_names: Optional list of class names
        title: Plot title
        figsize: Figure size (width, height)
        normalize: If True, interpret cm as normalized
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count" if not normalize else "Proportion"},
    )
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_crop_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    crop_indices: np.ndarray,
    output_dir: Path,
    crop_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plot confusion matrices for each crop.
    
    Args:
        predictions: Predicted class indices (N,)
        labels: Ground truth class indices (N,)
        crop_indices: Crop index for each sample (N,)
        output_dir: Directory to save figures
        crop_names: Optional crop names
        class_names: Optional class names
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_crops = np.unique(crop_indices)
    
    for crop_idx in unique_crops:
        mask = crop_indices == crop_idx
        crop_predictions = predictions[mask]
        crop_labels = labels[mask]
        
        cm = compute_confusion_matrix(crop_predictions, crop_labels, normalize=True)
        
        crop_name = crop_names[crop_idx] if crop_names else f"crop_{crop_idx}"
        output_path = output_dir / f"confusion_matrix_{crop_name}.png"
        
        plot_confusion_matrix(
            cm,
            output_path,
            class_names=class_names,
            title=f"Confusion Matrix: {crop_name}",
            normalize=True,
        )


def get_misclassified_classes(
    cm: np.ndarray,
    top_k: int = 5,
) -> List[tuple]:
    """
    Get top K most confused class pairs.
    
    Args:
        cm: Confusion matrix
        top_k: Number of top pairs to return
    
    Returns:
        List of (true_class, pred_class, count) tuples
    """
    misclassifications = []
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j:  # Off-diagonal
                count = cm[i, j]
                misclassifications.append((i, j, count))
    
    # Sort by count descending
    misclassifications.sort(key=lambda x: x[2], reverse=True)
    
    return misclassifications[:top_k]