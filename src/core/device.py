"""
Device and distributed training utilities.
"""

import torch
import os
from typing import Tuple


def get_device(rank: int = 0) -> torch.device:
    """
    Get device for current process in DDP setup.
    
    Args:
        rank: Process rank (from RANK env var or manual)
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    else:
        return torch.device("cpu")


def is_ddp() -> bool:
    """
    Check if running in DDP mode.
    
    Returns:
        True if RANK env var is set (DDP mode), False otherwise
    """
    return "RANK" in os.environ


def get_ddp_info() -> Tuple[int, int, int]:
    """
    Get DDP rank, world size, and local rank.
    Safe to call in non-DDP mode (returns defaults).
    
    Returns:
        (rank, world_size, local_rank)
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def is_main_process() -> bool:
    """
    Check if current process is rank 0 (main process).
    Use this to gate logging, checkpoint saving, etc.
    
    Returns:
        True if rank == 0, False otherwise
    """
    rank = int(os.environ.get("RANK", 0))
    return rank == 0