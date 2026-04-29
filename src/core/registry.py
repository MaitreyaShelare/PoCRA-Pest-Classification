"""
Registries for models, losses, optimizers, schedulers.
Allows flexible configuration from YAML.
"""

from typing import Dict, Callable, Any
import torch.nn as nn
import torch.optim as optim


class Registry:
    """Simple registry pattern for components."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Callable] = {}
    
    def register(self, key: str):
        """Decorator to register a class."""
        def wrapper(cls):
            self._items[key] = cls
            return cls
        return wrapper
    
    def get(self, key: str) -> Callable:
        """Get registered class."""
        if key not in self._items:
            raise KeyError(f"Unknown {self.name}: {key}")
        return self._items[key]
    
    def list(self) -> list:
        """List all registered items."""
        return list(self._items.keys())


# Create registries
model_registry = Registry("model")
loss_registry = Registry("loss")
optimizer_registry = Registry("optimizer")
scheduler_registry = Registry("scheduler")


# Register built-in optimizers
optimizer_registry.register("adam")(optim.Adam)
optimizer_registry.register("adamw")(optim.AdamW)
optimizer_registry.register("sgd")(optim.SGD)

# Register built-in schedulers
scheduler_registry.register("cosine")(optim.lr_scheduler.CosineAnnealingLR)
scheduler_registry.register("step")(optim.lr_scheduler.StepLR)
scheduler_registry.register("exponential")(optim.lr_scheduler.ExponentialLR)


def get_optimizer(cfg: Any, model: nn.Module) -> optim.Optimizer:
    """Create optimizer from config."""
    optimizer_class = optimizer_registry.get(cfg.optimizer)
    return optimizer_class(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def get_scheduler(cfg: Any, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
    """Create scheduler from config."""
    scheduler_class = scheduler_registry.get(cfg.scheduler)
    if cfg.scheduler == "cosine":
        return scheduler_class(optimizer, T_max=cfg.max_epochs)
    else:
        return scheduler_class(optimizer)