from typing import Dict, List


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    """
    return {"accuracy": 0.0}