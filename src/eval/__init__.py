"""Evaluation utilities: metrics, confusion matrix, calibration."""

from .metrics import (
    macro_f1,
    weighted_f1,
    accuracy,
    per_class_f1,
    per_class_accuracy,
    per_crop_accuracy,
    per_crop_f1,
    ood_auroc,
    ood_tpr_at_fpr,
    MetricTracker,
    compute_all_metrics,
)
from .confusion import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_per_crop_confusion_matrix,
    get_misclassified_classes,
)
from .calibration import (
    calibrate_ood_threshold,
    calibrate_class_confidence_threshold,
    sweep_prototype_distance_threshold,
    save_calibration_results,
)

__all__ = [
    # Metrics
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "per_class_f1",
    "per_class_accuracy",
    "per_crop_accuracy",
    "per_crop_f1",
    "ood_auroc",
    "ood_tpr_at_fpr",
    "MetricTracker",
    "compute_all_metrics",
    # Confusion
    "compute_confusion_matrix",
    "plot_confusion_matrix",
    "plot_per_crop_confusion_matrix",
    "get_misclassified_classes",
    # Calibration
    "calibrate_ood_threshold",
    "calibrate_class_confidence_threshold",
    "sweep_prototype_distance_threshold",
    "save_calibration_results",
]