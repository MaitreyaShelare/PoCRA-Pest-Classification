"""Early stopping utilities."""


class EarlyStopper:
    """Simple early stopping."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """
        Initialize early stopper.
        
        Args:
            patience: Epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = -float("inf")
    
    def __call__(self, current_metric: float) -> bool:
        """
        Check if should stop.
        
        Args:
            current_metric: Current validation metric (higher is better)
        
        Returns:
            True if should stop
        """
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience