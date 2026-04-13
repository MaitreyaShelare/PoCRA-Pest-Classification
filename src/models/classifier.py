"""
Model definitions.

Contains:
- backbone
- classification head

Does NOT contain training logic.
"""
import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    """
    Generic image classifier.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)