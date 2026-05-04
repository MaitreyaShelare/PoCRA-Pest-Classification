"""Learning rate scheduler utilities."""

import torch.optim as optim
from omegaconf import DictConfig


def get_scheduler(cfg: DictConfig, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
    """
    Create scheduler from config.
    
    Args:
        cfg: Phase config
        optimizer: Optimizer
    
    Returns:
        Scheduler instance
    """
    scheduler_name = cfg.scheduler.lower()
    
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_epochs,
            eta_min=1e-6,
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.get("step_size", 10),
            gamma=0.1,
        )
    elif scheduler_name == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")