"""
DDP utility functions: setup, cleanup, helpers.
"""

import torch
import torch.distributed as dist
import os
from typing import Tuple


def get_ddp_info() -> Tuple[int, int, int]:
    """
    Get DDP rank, world size, and local rank from environment.
    Safe to call in non-DDP mode (returns defaults).
    
    Returns:
        (rank, world_size, local_rank)
    
    Example:
        rank, world_size, local_rank = get_ddp_info()
        device = torch.device(f"cuda:{local_rank}")
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    return rank, world_size, local_rank


def is_ddp_enabled() -> bool:
    """Check if DDP is enabled (RANK env var exists)."""
    return "RANK" in os.environ


def print_once(message: str) -> None:
    """Print message from rank 0 only."""
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    
    if rank == 0:
        print(message)


def log_once(logger, message: str, level: str = "info") -> None:
    """Log message from rank 0 only."""
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    
    if rank == 0:
        getattr(logger, level)(message)


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a single tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        average: If True, compute mean; if False, sum
    
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    tensor = tensor.clone().cuda()
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    if average:
        tensor /= world_size
    
    return tensor


def get_local_rank() -> int:
    """Get local rank (rank within single node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_node_rank() -> int:
    """Get node rank (which node in multi-node setup)."""
    return int(os.environ.get("NODE_RANK", 0))


def barrier() -> None:
    """
    Synchronization barrier: all processes wait here.
    Useful before checkpoints or synchronized logging.
    """
    if dist.is_initialized():
        dist.barrier()