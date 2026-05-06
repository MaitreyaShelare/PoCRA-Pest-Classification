# 

"""Distributed training utilities for DDP."""

# Core DDP setup + process info
from .setup import (
    setup_ddp,
    cleanup_ddp,
    is_initialized,
    get_rank,
    get_world_size,
    is_main_process,
)

# DDP trainer wrapper
from .ddp_trainer import DDPTrainer

# Synchronization utilities
from .sync import (
    reduce_dict,
    gather_list,
    synchronize_between_processes,
)

# General distributed helper utilities
from .utils import (
    get_ddp_info,
    is_ddp_enabled,
    print_once,
    log_once,
    reduce_tensor,
    barrier,
    get_local_rank,
    get_node_rank,
    all_reduce,   # Added from Code 1
)

__all__ = [
    # =========================
    # Setup / Initialization
    # =========================
    "setup_ddp",
    "cleanup_ddp",
    "is_initialized",
    "get_rank",
    "get_world_size",
    "is_main_process",

    # =========================
    # Trainer wrapper
    # =========================
    "DDPTrainer",

    # =========================
    # Synchronization utilities
    # =========================
    "reduce_dict",
    "gather_list",
    "synchronize_between_processes",

    # =========================
    # DDP info / helpers
    # =========================
    "get_ddp_info",
    "is_ddp_enabled",
    "get_local_rank",
    "get_node_rank",

    # =========================
    # Logging / printing
    # =========================
    "print_once",
    "log_once",

    # =========================
    # Tensor synchronization
    # =========================
    "reduce_tensor",
    "all_reduce",

    # =========================
    # Process synchronization
    # =========================
    "barrier",
]