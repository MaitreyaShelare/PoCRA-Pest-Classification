"""
Inference pipeline for pest classification.
Handles single images and batches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np

from src.data.transforms import get_transforms
from src.distributed.utils import is_main_process


class PestClassificationPredictor:
    """
    End-to-end predictor for pest classification.
    Wraps the 5-stage pipeline for easy inference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        backbone: nn.Module,
        router_head: nn.Module,
        arcface_head: nn.Module,
        crop_heads_registry,
        crop_embeddings: nn.Embedding,
        crop_to_idx: Dict[str, int],
        pest_to_idx: Dict[str, int],
        type_to_idx: Dict[str, int],
        prototype_store: Optional[Dict] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.7,
        ood_threshold: float = 0.6,
    ) -> None:
        """
        Initialize predictor.
        
        Args:
            model: Full model or individual components
            backbone: DINOv2 backbone (frozen)
            router_head: Image type router
            arcface_head: ArcFace head
            crop_heads_registry: Per-crop heads
            crop_embeddings: Crop embeddings
            crop_to_idx: Mapping crop name -> index
            pest_to_idx: Mapping pest name -> index
            type_to_idx: Mapping image_type -> index
            prototype_store: Dict of pest prototypes for OOD detection
            device: Device to run on
            confidence_threshold: Min confidence to accept prediction
            ood_threshold: Threshold for OOD detection
        """
        self.backbone = backbone
        self.router_head = router_head
        self.arcface_head = arcface_head
        self.crop_heads_registry = crop_heads_registry
        self.crop_embeddings = crop_embeddings
        self.crop_to_idx = crop_to_idx
        self.pest_to_idx = pest_to_idx
        self.type_to_idx = type_to_idx
        self.prototype_store = prototype_store or {}
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ood_threshold = ood_threshold
        
        # Reverse mappings
        self.idx_to_crop = {v: k for k, v in crop_to_idx.items()}
        self.idx_to_pest = {v: k for k, v in pest_to_idx.items()}
        self.idx_to_type = {v: k for k, v in type_to_idx.items()}
        
        # Set to eval mode
        self.backbone.eval()
        self.router_head.eval()
        self.arcface_head.eval()
        for head in self.crop_heads_registry.pest_heads.values():
            head.eval()
        for head in self.crop_heads_registry.symptom_heads.values():
            head.eval()
    
    def predict_single(
        self,
        image: Union[Image.Image, str, Path],
        crop_name: str,
    ) -> Dict:
        """
        Predict on single image.
        
        Args:
            image: PIL Image, filepath, or Path
            crop_name: Crop name (e.g., "Cotton")
        
        Returns:
            Dict with predictions and metadata
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Invalid image type: {type(image)}")
        
        transform = get_transforms("val")
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get batch prediction and return first result
        results = self.predict_batch(image_tensor, [crop_name])
        return results[0]
    
    def predict_batch(
        self,
        images: torch.Tensor,
        crop_names: List[str],
    ) -> List[Dict]:
        """
        Predict on batch of images.
        
        Args:
            images: Image batch (B, 3, H, W), already preprocessed
            crop_names: List of crop names for each image (B,)
        
        Returns:
            List of prediction dicts
        """
        batch_size = images.shape[0]
        
        with torch.no_grad():
            # Stage 1: Route image type
            stage1_logits = self.router_head(self.backbone(images))
            image_type_probs = F.softmax(stage1_logits, dim=1)
            image_type_pred = stage1_logits.argmax(dim=1)
            image_type_conf = image_type_probs.max(dim=1).values
            
            # Stage 2: Extract pest embedding
            pest_embedding = self.backbone(images)
            pest_embedding = F.normalize(pest_embedding, p=2, dim=1)
            
            # Check OOD using prototypes
            ood_scores = self._compute_ood_scores(pest_embedding)
            
            # Stage 3: Route by image type
            results = []
            
            for i in range(batch_size):
                result = {
                    "crop": crop_names[i],
                    "image_type": self.idx_to_type.get(image_type_pred[i].item(), "unknown"),
                    "image_type_confidence": image_type_conf[i].item(),
                    "ood_score": ood_scores[i].item(),
                    "is_ood": ood_scores[i].item() < self.ood_threshold,
                }
                
                if result["is_ood"]:
                    result["pest"] = "UNKNOWN"
                    result["pest_confidence"] = 0.0
                    result["rejection_reason"] = "OOD: unknown pest species"
                
                elif image_type_pred[i].item() == 0:  # pest_body
                    # Stage 3A: Pest classification
                    crop_idx = torch.tensor(
                        [self.crop_to_idx[crop_names[i]]],
                        device=self.device,
                    )
                    crop_emb = self.crop_embeddings(crop_idx)
                    
                    pest_logits = self.crop_heads_registry.forward_pest(
                        crop_names[i],
                        pest_embedding[i:i+1],
                        crop_emb,
                    )
                    pest_probs = F.softmax(pest_logits, dim=1)
                    pest_conf = pest_probs.max().item()
                    pest_idx = pest_logits.argmax().item()
                    
                    result["pest"] = self.idx_to_pest.get(pest_idx, "unknown")
                    result["pest_confidence"] = pest_conf
                    
                    if pest_conf < self.confidence_threshold:
                        result["rejection_reason"] = f"low confidence: {pest_conf:.3f}"
                
                elif image_type_pred[i].item() == 1:  # symptom
                    # Stage 3B: Symptom classification
                    symptom_logits = self.crop_heads_registry.forward_symptom(
                        crop_names[i],
                        pest_embedding[i:i+1],
                    )
                    symptom_probs = F.softmax(symptom_logits, dim=1)
                    symptom_conf = symptom_probs.max().item()
                    symptom_idx = symptom_logits.argmax().item()
                    
                    result["symptom"] = self.idx_to_pest.get(symptom_idx, "unknown")
                    result["symptom_confidence"] = symptom_conf
                    
                    if symptom_conf < self.confidence_threshold:
                        result["rejection_reason"] = f"low confidence: {symptom_conf:.3f}"
                
                else:  # healthy
                    result["pest"] = "Healthy"
                    result["pest_confidence"] = image_type_conf[i].item()
                
                results.append(result)
        
        return results
    
    def _compute_ood_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute OOD scores using prototype distance.
        Lower score = more OOD.
        
        Args:
            embeddings: Normalized embeddings (B, 768)
        
        Returns:
            OOD scores (B,) - higher = more confident ID
        """
        if not self.prototype_store:
            # No prototypes, assume ID
            return torch.ones(embeddings.shape[0], device=embeddings.device)
        
        # Compute max distance to any pest prototype
        max_distances = []
        
        for pest_name, proto_dict in self.prototype_store.items():
            # Check both lateral and overhead prototypes
            dists = []
            for vp in ["lateral", "overhead"]:
                proto = proto_dict.get(vp)
                if proto is not None:
                    proto = proto.to(embeddings.device)
                    # Cosine distance
                    dist = torch.mm(embeddings, proto.unsqueeze(1)).squeeze(1)
                    dists.append(dist)
            
            if dists:
                # Max similarity across viewpoints
                max_dist = torch.stack(dists).max(dim=0).values
                max_distances.append(max_dist)
        
        if max_distances:
            # OOD score: min distance to any known pest
            ood_score = torch.stack(max_distances).min(dim=0).values
        else:
            ood_score = torch.ones(embeddings.shape[0], device=embeddings.device)
        
        return ood_score


def predict_single(
    image: Union[Image.Image, str, Path],
    crop_name: str,
    checkpoint_path: Path,
    device: str = "cuda",
) -> Dict:
    """
    Quick predict on single image from checkpoint.
    
    Args:
        image: Image input
        crop_name: Crop name
        checkpoint_path: Path to checkpoint
        device: Device
    
    Returns:
        Prediction dict
    """
    # Load checkpoint and reconstruct model
    # (Simplified - full implementation would load full model state)
    # For now, user would construct predictor manually
    
    raise NotImplementedError(
        "Use PestClassificationPredictor class with loaded model instead"
    )


def predict_directory(
    image_dir: Path,
    crop_name: str,
    predictor: PestClassificationPredictor,
    batch_size: int = 32,
    output_csv: Optional[Path] = None,
) -> List[Dict]:
    """
    Predict on all images in directory.
    
    Args:
        image_dir: Directory with images
        crop_name: Crop name
        predictor: PestClassificationPredictor instance
        batch_size: Batch size
        output_csv: Optional path to save results
    
    Returns:
        List of prediction dicts
    """
    from torch.utils.data import DataLoader, Dataset
    import csv
    from tqdm import tqdm
    
    class ImageDataset(Dataset):
        def __init__(self, image_dir: Path):
            self.images = list(image_dir.glob("*.*"))
            self.transform = get_transforms("val")
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert("RGB")
            return self.transform(img), str(self.images[idx])
    
    dataset = ImageDataset(image_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_results = []
    
    with torch.no_grad():
        for batch_imgs, filepaths in tqdm(loader):
            crop_names = [crop_name] * len(batch_imgs)
            batch_results = predictor.predict_batch(batch_imgs, crop_names)
            
            for result, filepath in zip(batch_results, filepaths):
                result["filepath"] = filepath
                all_results.append(result)
    
    # Save to CSV if requested
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    return all_results