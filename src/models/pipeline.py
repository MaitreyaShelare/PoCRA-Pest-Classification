"""
Complete 5-stage inference pipeline.
Coordinates all stages from segmentation to final classification.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from pathlib import Path

from src.models.backbones.dinov2 import DINOv2Backbone
from src.models.heads.arcface_head import ArcFaceHead
from src.models.heads.router_head import RouterHead
from src.models.heads.crop_heads import CropHeadsRegistry
from src.models.sam import SAMSegmenter


class PestClassificationPipeline(nn.Module):
    """
    Complete 5-stage pest classification pipeline.
    
    Stage 0: Segmentation (SAM)
    Stage 1: Image type routing
    Stage 2: Fine-grained pest embedding (ArcFace)
    Stage 3A: Crop-conditioned pest classification
    Stage 3B: Crop-conditioned symptom classification
    """
    
    def __init__(
        self,
        backbone: DINOv2Backbone,
        router_head: RouterHead,
        arcface_head: ArcFaceHead,
        crop_heads: CropHeadsRegistry,
        crop_embeddings: nn.Embedding,
        prototype_store: Dict = None,
        ood_thresholds: Dict = None,
        device: str = "cuda",
    ) -> None:
        """
        Initialize pipeline.
        
        Args:
            backbone: DINOv2 backbone
            router_head: Image type router
            arcface_head: ArcFace head
            crop_heads: Per-crop heads registry
            crop_embeddings: Learned crop embeddings
            prototype_store: Dual prototypes per pest class
            ood_thresholds: OOD rejection thresholds
            device: Device to run on
        """
        super().__init__()
        
        self.backbone = backbone
        self.router_head = router_head
        self.arcface_head = arcface_head
        self.crop_heads = crop_heads
        self.crop_embeddings = crop_embeddings
        self.prototype_store = prototype_store or {}
        self.ood_thresholds = ood_thresholds or {}
        self.device = device
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        crop_label: str,
    ) -> Dict:
        """
        Full pipeline inference.
        
        Args:
            images: Image batch (B, 3, 224, 224)
            crop_label: Crop name (e.g., "Cotton")
        
        Returns:
            Dict with predictions and metadata
        """
        B = images.shape[0]
        device = images.device
        
        # Stage 1: Route image type
        stage1_logits = self.router_head(self.backbone(images))
        image_type_probs = torch.softmax(stage1_logits, dim=1)
        image_type_pred = stage1_logits.argmax(dim=1)
        image_type_conf = image_type_probs.max(dim=1).values
        
        # Stage 2: Extract pest embedding
        pest_embedding = self.backbone(images)
        pest_embedding = torch.nn.functional.normalize(pest_embedding, p=2, dim=1)
        
        # Stage 3: Route based on image type
        results = {
            "image_type": ["pest_body", "symptom", "healthy"][image_type_pred[0].item()],
            "image_type_confidence": image_type_conf.mean().item(),
        }
        
        if image_type_pred[0] == 0:  # pest_body
            # Stage 3A: Pest classification
            crop_emb = self.crop_embeddings(torch.tensor([self.crop_to_idx[crop_label]], device=device))
            pest_logits = self.crop_heads.forward_pest(crop_label, pest_embedding, crop_emb)
            pest_probs = torch.softmax(pest_logits, dim=1)
            results["pest"] = pest_logits.argmax(dim=1).item()
            results["pest_confidence"] = pest_probs.max().item()
        
        elif image_type_pred[0] == 1:  # symptom
            # Stage 3B: Symptom classification
            symptom_logits = self.crop_heads.forward_symptom(crop_label, pest_embedding)
            symptom_probs = torch.softmax(symptom_logits, dim=1)
            results["symptom"] = symptom_logits.argmax(dim=1).item()
            results["symptom_confidence"] = symptom_probs.max().item()
        
        else:  # healthy
            results["pest"] = "Healthy"
            results["pest_confidence"] = image_type_conf.mean().item()
        
        return results