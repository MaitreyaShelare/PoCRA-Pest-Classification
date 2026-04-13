import torch
from torch import nn


def save_checkpoint(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    model.load_state_dict(torch.load(path))
    return model