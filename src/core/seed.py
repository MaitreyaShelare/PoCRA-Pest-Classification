"""
Global seed initialization for reproducibility across the codebase.
"""

import random
import numpy as np
import torch
from typing import Optional


_GLOBAL_SEED: Optional[int] = None


def seed_everything(seed: int) -> None:
    """
    Set seed globally in one call.
    Call this at the start of training.
    
    Args:
        seed: Seed value
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_global_seed() -> Optional[int]:
    """Get currently set global seed."""
    return _GLOBAL_SEED