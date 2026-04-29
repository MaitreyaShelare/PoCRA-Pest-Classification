"""Distributed training utilities for DDP."""

from .setup import (
    setup_ddp,
    cleanup_ddp,
    is_initialized,
    get_rank,
    get_world_size,
    is_main_process,
)
from .ddp_trainer import DDPTrainer
from .sync import reduce_dict, gather_list, synchronize_between_processes
from .utils import (
    get_ddp_info,
    is_ddp_enabled,
    print_once,
    log_once,
    reduce_tensor,
    barrier,
    get_local_rank,
    get_node_rank,
)

__all__ = [
    # Setup
    "setup_ddp",
    "cleanup_ddp",
    "is_initialized",
    "get_rank",
    "get_world_size",
    "is_main_process",
    # Trainer wrapper
    "DDPTrainer",
    # Metrics sync
    "reduce_dict",
    "gather_list",
    "synchronize_between_processes",
    # Utils
    "get_ddp_info",
    "is_ddp_enabled",
    "print_once",
    "log_once",
    "reduce_tensor",
    "barrier",
    "get_local_rank",
    "get_node_rank",
]