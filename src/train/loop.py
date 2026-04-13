"""
Trainer module.

Responsible for:
- training loop
- validation loop
- logging
- checkpointing

Does NOT define model architecture.
"""
import torch
from torch.utils.data import DataLoader
from torch import nn


class Trainer:
    """
    Handles training and validation.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss: float = 0.0

        for x, y in loader:
            x = x
            y = y

            output: torch.Tensor = self.model(x)
            loss: torch.Tensor = output.mean()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)