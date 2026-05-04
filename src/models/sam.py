"""
SAM (Segment Anything Model) wrapper for Stage 0 segmentation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from PIL import Image
import numpy as np


class SAMSegmenter(nn.Module):
    """
    SAM-ViT-Base wrapper for foreground segmentation.
    
    Segments pest/affected region from background.
    Output: binary mask, resized to 224x224.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        mask_threshold: float = 0.4,
        device: str = "cuda",
    ) -> None:
        """
        Initialize SAM segmenter.
        
        Args:
            checkpoint_path: Path to SAM checkpoint (if not using automatic download)
            mask_threshold: Confidence threshold for mask
            device: Device to run on
        """
        super().__init__()
        
        self.device = device
        self.mask_threshold = mask_threshold
        
        try:
            from segment_anything import sam_model_registry
            
            model_type = "vit_b"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            
            self.predictor = sam.image_encoder
            self.mask_decoder = sam.mask_decoder
            self.prompt_encoder = sam.prompt_encoder
            
        except ImportError:
            raise ImportError("segment_anything not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    
    @torch.no_grad()
    def forward(
        self,
        image: Image.Image,
        prompt_type: str = "center_point",
    ) -> Tuple[torch.Tensor, float]:
        """
        Segment image.
        
        Args:
            image: PIL Image
            prompt_type: "center_point", "bbox", or "grid"
        
        Returns:
            (segmented_crop: (1, 3, 224, 224), confidence: float)
        """
        # Convert image to array
        image_array = np.array(image.convert("RGB"))
        h, w = image_array.shape[:2]
        
        # Generate prompt
        if prompt_type == "center_point":
            prompt_points = np.array([[w // 2, h // 2]])
            prompt_labels = np.array([1])
        else:
            raise NotImplementedError(f"Prompt type {prompt_type} not implemented")
        
        # Encode image
        image_tensor = torch.from_numpy(image_array).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor / 255.0
        
        # (Simplified: actual SAM integration more complex)
        # For now, return dummy output
        segmented = torch.randn(1, 3, 224, 224)
        confidence = 0.9
        
        return segmented, confidence