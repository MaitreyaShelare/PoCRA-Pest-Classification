"""
DataLoader creation with proper settings for training/validation/testing.
Handles distributed sampling, worker seeds, etc.
"""

from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing import Optional
import os


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    split: str = "train",
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Create a DataLoader with appropriate settings.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        split: "train", "val", or "test" (for auto shuffle)
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer
        drop_last: Drop last incomplete batch (for training)
        shuffle: Explicit shuffle setting (auto if None)
    
    Returns:
        DataLoader
    """
    
    # Auto-determine shuffle
    if shuffle is None:
        shuffle = (split == "train")
    
    # Distributed sampling (if in DDP mode)
    sampler = None
    # if "RANK" in os.environ:
    #     world_size = int(os.environ.get("WORLD_SIZE", 1))
    #     rank = int(os.environ.get("RANK", 0))
    #     sampler = DistributedSampler(
    #         dataset,
    #         num_replicas=world_size,
    #         rank=rank,
    #         shuffle=shuffle,
    #         seed=42,  # Use config seed if available
    #     )
    #     shuffle = False  # DistributedSampler handles shuffle
    if "RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=42,
            drop_last=True,
        )

        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last and (split == "train"),
        persistent_workers=(num_workers > 0),
    )


# def get_train_dataloader(
#     dataset: Dataset,
#     batch_size: int,
#     num_workers: int = 4,
# ) -> DataLoader:
#     """Convenience function for training dataloader."""
#     return get_dataloader(
#         dataset,
#         batch_size=batch_size,
#         split="train",
#         num_workers=num_workers,
#         drop_last=True,
#     )
def get_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    drop_last: bool = True,
) -> DataLoader:
    """Convenience function for training dataloader."""

    return get_dataloader(
        dataset,
        batch_size=batch_size,
        split="train",
        num_workers=num_workers,
        drop_last=drop_last,
    )

def get_val_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Convenience function for validation dataloader."""
    return get_dataloader(
        dataset,
        batch_size=batch_size,
        split="val",
        num_workers=num_workers,
        drop_last=False,
    )


def get_test_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Convenience function for test dataloader."""
    return get_dataloader(
        dataset,
        batch_size=batch_size,
        split="test",
        num_workers=num_workers,
        drop_last=False,
    )