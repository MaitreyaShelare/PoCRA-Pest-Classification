"""
Base classes for models and heads.
Defines interfaces that all components must implement.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseBackbone(nn.Module, ABC):
    """Abstract backbone class."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image.
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            Feature tensor (B, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get feature embedding dimension."""
        pass
    
    @abstractmethod
    def freeze(self) -> None:
        """Freeze all parameters."""
        pass
    
    @abstractmethod
    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        pass


class BaseHead(nn.Module, ABC):
    """Abstract head class."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute head output.
        
        Args:
            x: Input features (B, in_features)
        
        Returns:
            Output tensor (B, num_classes) or (B, embedding_dim)
        """
        pass


class BaseLoss(nn.Module, ABC):
    """Abstract loss class."""
    
    @abstractmethod
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            input: Model output (B, num_classes) or (B, embedding_dim)
            target: Target labels (B,)
        
        Returns:
            Scalar loss
        """
        pass