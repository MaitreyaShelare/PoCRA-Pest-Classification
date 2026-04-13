import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Run inference.
    """
    model.eval()
    with torch.no_grad():
        return model(x)