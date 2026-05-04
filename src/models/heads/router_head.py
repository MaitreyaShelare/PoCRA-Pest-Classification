"""
Router head for Stage 1: classify image type (pest body / symptom / healthy).
"""

import torch
import torch.nn as nn
from src.models.base import BaseHead


class RouterHead(BaseHead):
    """
    Image type classifier for routing to downstream stages.
    
    3-way classification: pest_body, symptom, healthy
    """
    
    def __init__(
        self,
        in_features: int = 768,
        num_types: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize router head.
        
        Args:
            in_features: Input feature dimension (768 for DINOv2)
            num_types: Number of image types (3: pest_body, symptom, healthy)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_types = num_types
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, num_types),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify image type.
        
        Args:
            x: Input features (B, in_features)
        
        Returns:
            Logits (B, num_types)
        """
        return self.classifier(x)


class RouterLoss(nn.Module):
    """Router head + cross-entropy loss."""
    
    def __init__(
        self,
        in_features: int = 768,
        num_types: int = 3,
    ) -> None:
        """Initialize router loss."""
        super().__init__()
        self.head = RouterHead(in_features=in_features, num_types=num_types)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute router loss.
        
        Args:
            features: Input features (B, in_features)
            labels: Image type labels (B,) - 0: pest_body, 1: symptom, 2: healthy
        
        Returns:
            Scalar loss
        """
        logits = self.head(features)
        loss = self.loss_fn(logits, labels)
        return loss