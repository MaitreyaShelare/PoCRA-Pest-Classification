"""
Trainer module.

Responsible for:
- epoch orchestration
- calling train/val loops
- logging
- checkpointing

Does NOT contain batch-level logic.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from .loop import train_one_epoch, validate


class Trainer:
    """
    High-level training controller.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> None:
        self.model: nn.Module = model.to(device)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion: nn.Module = criterion
        self.device: torch.device = device

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> None:
        """
        Runs full training.

        Args:
            train_loader: training data
            val_loader: validation data
            epochs: number of epochs
        """
        for epoch in range(epochs):
            train_loss: float = train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
            )

            val_loss: float = validate(
                model=self.model,
                loader=val_loader,
                criterion=self.criterion,
                device=self.device,
            )

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f}"
            )