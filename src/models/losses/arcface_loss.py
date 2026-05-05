"""
ArcFace Loss for metric learning.
Combines angular margin with cross-entropy for tight species-level clustering.
Paper: https://arxiv.org/abs/1801.07698
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from src.models.base import BaseLoss


class ArcFaceHead(nn.Module):
    """
    ArcFace head: maps embeddings to class logits with angular margin.
    
    Key idea:
    - Normalize embeddings and weight matrix
    - Compute cosine similarity (angle between vectors)
    - Apply angular margin: θ + m
    - Scale logits for numerical stability
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
            in_features: Input embedding dimension (e.g., 768 for DINOv2 ViT-B/14)
            num_classes: Number of classes (pest species)
            margin: Angular margin in radians
                   - Small margin (0.3-0.5): softer boundaries
                   - Large margin (0.5-1.0): harder boundaries
                   Typical: 0.5
            scale: Scaling factor for logits
                   - Higher scale: sharper softmax
                   - Typical: 64
            easy_margin: If True, skip margin for easy samples (cosine > 0)
                        Helps with very imbalanced data
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        
        # Weight matrix: class centers in embedding space
        # Shape: (num_classes, in_features)
        # Each row is a class prototype
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute trigonometric values for numerical stability
        # cos(θ + m) = cos(θ)*cos(m) - sin(θ)*sin(m)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        
        # Threshold: cos(π - m)
        # Used for numerical stability check
        self.th = math.cos(math.pi - margin)
        
        # Fallback: sin(π - m) * m
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace logits with margin.
        
        Args:
            input: Embedding tensor (B, in_features)
                  Should be L2-normalized: norm(input) ≈ 1
            label: Class labels (B,), values in [0, num_classes)
        
        Returns:
            Logits with margin applied (B, num_classes)
        """
        # Normalize input and weight for cosine similarity
        # This ensures we're working in angle space (not magnitude space)
        input_norm = F.normalize(input, p=2, dim=1)      # (B, in_features)
        weight_norm = F.normalize(self.weight, p=2, dim=1)  # (num_classes, in_features)
        
        # Compute cosine similarity: cos(θ) = <input, weight> / (||input|| * ||weight||)
        # Since both are normalized, this is just the dot product
        cosine = F.linear(input_norm, weight_norm)  # (B, num_classes)
        
        # Clamp to [-1, 1] to handle numerical errors
        cosine = torch.clamp(cosine, -1.0, 1.0)
        
        # Compute sine from cosine: sin(θ) = sqrt(1 - cos²(θ))
        # Use clamp to avoid NaN from negative values
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # Apply angular margin: cos(θ + m) = cos(θ)*cos(m) - sin(θ)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Handle numerical stability and easy_margin flag
        if self.easy_margin:
            # For easy samples (cosine > 0), use original cosine
            # This avoids margin for samples already far from decision boundary
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard ArcFace: use original cosine when θ + m > π
            # This prevents θ + m from exceeding π (numerical instability)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoded target labels
        # one_hot[i, label[i]] = 1, else 0
        one_hot = torch.zeros(cosine.size(), device=cosine.device, dtype=cosine.dtype)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Blend: use phi (with margin) for target class, cosine for others
        # This applies margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale logits for numerical stability in softmax
        # Scaling typically sharpens the softmax (higher scale = more confident decisions)
        output *= self.scale
        
        return output


class ArcFaceLoss(BaseLoss):
    """
    Complete ArcFace loss: ArcFace head + cross-entropy.
    
    Pipeline:
    1. Normalize embeddings
    2. Apply ArcFace margin (angular)
    3. Compute cross-entropy on scaled logits
    
    This loss encourages:
    - Tight clustering within classes (margin enforces separation)
    - Diverse embeddings (normalized feature space)
    - Discriminative boundaries (cross-entropy)
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        easy_margin: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Initialize ArcFace loss.
        
        Args:
            in_features: Embedding dimension
            num_classes: Number of classes
            margin: Angular margin
            scale: Scaling factor
            easy_margin: Skip margin for easy samples
            label_smoothing: Label smoothing for cross-entropy (helps with overconfidence)
        """
        super().__init__()
        
        self.head = ArcFaceHead(
            in_features=in_features,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
            easy_margin=easy_margin,
        )
        
        # Cross-entropy loss (standard classification loss)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: Embedding tensor (B, in_features)
                       Will be normalized internally
            labels: Class labels (B,)
        
        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Apply ArcFace margin
        logits = self.head(embeddings_norm, labels)
        
        # Compute cross-entropy loss on margined logits
        loss = self.loss_fn(logits, labels)
        
        return loss


class ArcFaceWithSupConLoss(nn.Module):
    """
    Combined ArcFace + Supervised Contrastive Loss.
    Used in Phase 1 training for better clustering.
    
    ArcFace: discriminative loss (angular margin)
    SupCon: contrastive loss (tight within-class clusters)
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        arcface_weight: float = 1.0,
        supcon_weight: float = 0.5,
        supcon_temperature: float = 0.07,
    ) -> None:
        """
        Initialize combined loss.
        
        Args:
            in_features: Embedding dimension
            num_classes: Number of classes
            margin: ArcFace angular margin
            scale: ArcFace scale
            arcface_weight: Weight for ArcFace loss
            supcon_weight: Weight for SupCon loss
            supcon_temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.arcface_loss = ArcFaceLoss(
            in_features=in_features,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )
        
        # Import SupCon loss
        from src.models.losses.supcon_loss import SupConLoss
        self.supcon_loss = SupConLoss(temperature=supcon_temperature)
        
        self.arcface_weight = arcface_weight
        self.supcon_weight = supcon_weight
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            embeddings: Embeddings (B, in_features)
            labels: Class labels (B,)
        
        Returns:
            (total_loss, arcface_loss, supcon_loss)
        """
        # Normalize
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # ArcFace loss
        arcface_loss = self.arcface_loss(embeddings_norm, labels)
        
        # SupCon loss (reshape for contrastive learning)
        supcon_loss = self.supcon_loss(
            embeddings_norm.unsqueeze(1),  # (B, 1, in_features)
            labels=labels,
        )
        
        # Combined loss
        total_loss = (
            self.arcface_weight * arcface_loss +
            self.supcon_weight * supcon_loss
        )
        
        return total_loss, arcface_loss, supcon_loss


# Example usage for Phase 1 training:
"""
loss_fn = ArcFaceWithSupConLoss(
    in_features=768,
    num_classes=num_species,
    margin=0.5,
    scale=64.0,
    arcface_weight=1.0,
    supcon_weight=0.5,
)

for batch in train_loader:
    images, labels = batch
    embeddings = backbone(images)  # (B, 768)
    
    total_loss, arcface_loss, supcon_loss = loss_fn(embeddings, labels)
    
    total_loss.backward()
    optimizer.step()
"""