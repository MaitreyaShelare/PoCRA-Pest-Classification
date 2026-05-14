"""
DINOv2 ViT-B/14 backbone for pest classification.
Pretrained on large-scale unlabeled data, excellent for fine-grained tasks.
"""

import torch
import torch.nn as nn
from typing import Optional
from src.models.base import BaseBackbone
from src.distributed.utils import print_once


class DINOv2Backbone(BaseBackbone):
    """
    DINOv2 ViT-B/14 backbone.
    
    Loads pretrained ViT-B/14 from torch.hub.
    Extracts CLS token as image embedding (768-dim).
    Can be frozen or fine-tuned.
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        embedding_dim: int = 768,
        freeze_backbone: bool = False,
    ) -> None:
        """
        Initialize DINOv2 backbone.
        
        Args:
            model_name: Model name from torch.hub (dinov2_vitb14, dinov2_vits14, etc.)
            embedding_dim: Output embedding dimension (should match model)
            freeze_backbone: If True, freeze parameters immediately
        """
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._freeze_backbone = freeze_backbone
        
        # Load pretrained model
        print_once(f"  Loading {model_name} from torch.hub...")
        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=True,
            )
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            print(f"  Make sure you have internet access or cache the model locally")
            raise
        
        # Verify embedding dimension
        if embedding_dim != 768:
            raise ValueError(
                f"ViT-B/14 has embedding_dim=768, got {embedding_dim}"
            )
        
        # Freeze if requested
        if freeze_backbone:
            self.freeze()
        else:
            self.unfreeze()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image.
        
        Args:
            x: Image tensor (B, 3, H, W), normalized to ImageNet stats
        
        Returns:
            Feature tensor (B, 768) - CLS token embedding
        """
        with torch.set_grad_enabled(not self._freeze_backbone):
            # Forward through backbone
            features = self.backbone(x)  # (B, 768)
        
        return features
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    # def freeze(self) -> None:
    #     """Freeze all parameters."""
    #     self._freeze_backbone = True
    #     for param in self.backbone.parameters():
    #         param.requires_grad = False
    def freeze(self) -> None:
        """Freeze all parameters."""
        
        self._freeze_backbone = True

        module = (
            self.backbone.module
            if hasattr(self.backbone, "module")
            else self.backbone
        )

        for param in module.parameters():
            param.requires_grad = False

    # def unfreeze(self) -> None:
    #     """Unfreeze all parameters."""
    #     self._freeze_backbone = False
    #     for param in self.backbone.parameters():
    #         param.requires_grad = True

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        
        self._freeze_backbone = False

        module = (
            self.backbone.module
            if hasattr(self.backbone, "module")
            else self.backbone
        )

        for param in module.parameters():
            param.requires_grad = True
    
    def is_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return self._freeze_backbone