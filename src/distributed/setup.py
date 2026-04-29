"""
Distributed Data Parallel (DDP) setup and teardown.
Initialize DDP at start of training, cleanup at end.
"""

import torch
import torch.distributed as dist
import os
from typing import Tuple


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialize DDP process group.
    Call this at the start of each rank's training process.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: "nccl" for GPU, "gloo" for CPU
    
    Example:
        rank, world_size, local_rank = get_ddp_info()
        setup_ddp(rank, world_size)
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.timedelta(minutes=30),
    )
    
    if rank == 0:
        print(f"DDP initialized: rank {rank}/{world_size}, backend={backend}")


def cleanup_ddp() -> None:
    """
    Destroy DDP process group.
    Call this at the end of training (in finally block or atexit).
    
    Example:
        try:
            train_loop()
        finally:
            cleanup_ddp()
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_initialized() -> bool:
    """Check if DDP is initialized."""
    return dist.is_initialized()


def get_rank() -> int:
    """Get current process rank. Safe to call even if DDP not initialized."""
    if is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes. Safe to call even if DDP not initialized."""
    if is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is rank 0 (main process)."""
    return get_rank() == 0