"""
Exponential Moving Average (EMA) for model parameters.
Optional: use in Phase 1 for smoother training.
"""

import torch
import torch.nn as nn
from typing import Optional


class EMA:
    """
    Exponential Moving Average of model parameters.
    Useful for stabilizing training.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        """
        Initialize EMA.
        
        Args:
            model: Model to track
            decay: EMA decay factor (typical: 0.999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        
        # Create shadow copy
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_shadow = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_shadow
    
    def apply_shadow(self) -> None:
        """Apply shadow to model (for validation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()
    
    def restore(self) -> None:
        """Restore original parameters."""
        # Would need to track originals separately
        pass