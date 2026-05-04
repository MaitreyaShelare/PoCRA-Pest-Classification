"""
Per-crop pest and symptom classification heads (Stage 3A and 3B).
Registry pattern for managing multiple crop-specific heads.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class CropPestHead(nn.Module):
    """Stage 3A: crop-specific pest body classifier with crop conditioning."""
    
    def __init__(
        self,
        in_features: int = 768,
        crop_embedding_dim: int = 128,
        num_pests: int = None,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize crop pest head.
        
        Args:
            in_features: Pest embedding dimension (768)
            crop_embedding_dim: Crop embedding dimension (128)
            num_pests: Number of pests for this crop
            mlp_hidden_dim: MLP hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        concat_dim = in_features + crop_embedding_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(concat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_pests),
        )
    
    def forward(
        self,
        pest_embedding: torch.Tensor,
        crop_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify pest.
        
        Args:
            pest_embedding: Stage 2 embedding (B, 768)
            crop_embedding: Crop embedding (B, 128)
        
        Returns:
            Logits (B, num_pests)
        """
        concat = torch.cat([pest_embedding, crop_embedding], dim=1)
        return self.classifier(concat)


class CropSymptomHead(nn.Module):
    """Stage 3B: crop-specific symptom classifier."""
    
    def __init__(
        self,
        in_features: int = 768,
        num_symptoms: int = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize crop symptom head.
        
        Args:
            in_features: Backbone feature dimension (768)
            num_symptoms: Number of symptom types for this crop
            dropout: Dropout probability
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, num_symptoms),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify symptom.
        
        Args:
            features: Backbone features (B, 768)
        
        Returns:
            Logits (B, num_symptoms)
        """
        return self.classifier(features)


class CropHeadsRegistry:
    """Registry for managing per-crop pest and symptom heads."""
    
    def __init__(
        self,
        crops: list,
        num_pests_per_crop: Dict[str, int],
        num_symptoms_per_crop: Dict[str, int],
        in_features: int = 768,
        crop_embedding_dim: int = 128,
        mlp_hidden_dim: int = 256,
    ) -> None:
        """
        Initialize crop heads registry.
        
        Args:
            crops: List of crop names
            num_pests_per_crop: Dict mapping crop -> num_pests
            num_symptoms_per_crop: Dict mapping crop -> num_symptoms
            in_features: Embedding dimension
            crop_embedding_dim: Crop embedding dimension
            mlp_hidden_dim: MLP hidden dimension
        """
        self.crops = crops
        self.in_features = in_features
        self.crop_embedding_dim = crop_embedding_dim
        
        # Create heads for each crop
        self.pest_heads = nn.ModuleDict()
        self.symptom_heads = nn.ModuleDict()
        
        for crop in crops:
            n_pests = num_pests_per_crop.get(crop, 0)
            n_symptoms = num_symptoms_per_crop.get(crop, 0)
            
            if n_pests > 0:
                self.pest_heads[crop] = CropPestHead(
                    in_features=in_features,
                    crop_embedding_dim=crop_embedding_dim,
                    num_pests=n_pests,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
            
            if n_symptoms > 0:
                self.symptom_heads[crop] = CropSymptomHead(
                    in_features=in_features,
                    num_symptoms=n_symptoms,
                )
    
    def forward_pest(
        self,
        crop: str,
        pest_embedding: torch.Tensor,
        crop_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify pest for given crop.
        
        Args:
            crop: Crop name
            pest_embedding: Stage 2 embedding (B, 768)
            crop_embedding: Crop embedding (B, 128)
        
        Returns:
            Logits (B, num_pests_for_crop)
        """
        if crop not in self.pest_heads:
            raise ValueError(f"Unknown crop: {crop}")
        return self.pest_heads[crop](pest_embedding, crop_embedding)
    
    def forward_symptom(
        self,
        crop: str,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify symptom for given crop.
        
        Args:
            crop: Crop name
            features: Backbone features (B, 768)
        
        Returns:
            Logits (B, num_symptoms_for_crop)
        """
        if crop not in self.symptom_heads:
            raise ValueError(f"Unknown crop: {crop}")
        return self.symptom_heads[crop](features)