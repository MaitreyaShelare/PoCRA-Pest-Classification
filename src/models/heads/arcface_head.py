"""
ArcFace head for metric learning.
Learns tight species-level pest clusters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.base import BaseHead


class ArcFaceHead(BaseHead):
    """
    ArcFace margin head for metric learning.
    
    Maps embedding to logits using angular margin.
    Enforces tight within-class and wide between-class separation.
    
    Paper: https://arxiv.org/abs/1801.07698
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        easy_margin: bool = False,
    ) -> None:
        """
        Initialize ArcFace head.
        
        Args:
            in_features: Input embedding dimension (768 for DINOv2 ViT-B/14)
            num_classes: Number of classes (pest species)
            margin: Angular margin in radians (typical: 0.5)
            scale: Scaling factor for logits (typical: 64)
            easy_margin: If True, skip margin for easy samples (cosine > 0)
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        
        # Weight matrix (class centers in embedding space)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute margin values for numerical stability
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace logits.
        
        Args:
            input: Normalized embedding (B, in_features)
            target: Class labels (B,)
        
        Returns:
            Logits with margin applied (B, num_classes)
        """
        # Normalize input and weight
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(input_norm, weight_norm)  # (B, num_classes)
        
        # Compute sine from cosine: sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # Apply margin: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Numerical stability: if easy_margin, use original cosine for safe samples
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard ArcFace: use original cosine if cos(theta) < cos(pi - m)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot target
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        
        # Blend: use phi for target class, cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale by s
        output *= self.scale
        
        return output


class ArcFaceLoss(nn.Module):
    """Combined ArcFace head + cross-entropy loss."""
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ) -> None:
        """
        Initialize ArcFace loss.
        
        Args:
            in_features: Embedding dimension
            num_classes: Number of classes
            margin: Angular margin
            scale: Scale factor
        """
        super().__init__()
        self.head = ArcFaceHead(
            in_features=in_features,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: Embeddings (B, in_features), should be normalized
            labels: Class labels (B,)
        
        Returns:
            Scalar loss
        """
        logits = self.head(embeddings, labels)
        loss = self.loss_fn(logits, labels)
        return loss