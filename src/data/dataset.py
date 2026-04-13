"""
Dataset module.

Responsible for:
- loading images
- returning (image, label)

Does NOT handle batching or training.
"""

from torch.utils.data import Dataset
import torch
from typing import Tuple


class ImageDataset(Dataset):
    """
    Dataset for image classification.
    Returns (image, label).
    """

    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError