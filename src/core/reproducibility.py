"""
Reproducibility utilities: seed setting, deterministic execution.
Critical for research reproducibility.
"""

import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int) -> None:
    """
    Set seed for all random number generators.
    Makes training deterministic and reproducible.
    
    Args:
        seed: Seed value to use everywhere
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDA determinism
    # WARNING: This may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Seed function for DataLoader workers.
    Ensures each worker gets different random state.
    
    Usage:
        loader = DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            ...
        )
    
    Args:
        worker_id: Worker ID (automatic from DataLoader)
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)