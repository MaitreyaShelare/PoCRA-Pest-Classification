"""Optimizer utilities."""

import torch.optim as optim
from omegaconf import DictConfig


def get_optimizer(cfg: DictConfig, model) -> optim.Optimizer:
    """
    Create optimizer from config.
    
    Args:
        cfg: Phase config
        model: Model or model parameters
    
    Returns:
        Optimizer instance
    """
    optimizer_name = cfg.optimizer.lower()
    lr = cfg.learning_rate
    wd = cfg.weight_decay
    
    if optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters() if hasattr(model, 'parameters') else model,
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
        )
    elif optimizer_name == "adam":
        return optim.Adam(
            model.parameters() if hasattr(model, 'parameters') else model,
            lr=lr,
            weight_decay=wd,
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters() if hasattr(model, 'parameters') else model,
            lr=lr,
            weight_decay=wd,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")