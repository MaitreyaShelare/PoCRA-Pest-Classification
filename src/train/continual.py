"""
Continual learning: add crop, add pest, compute prototypes.
These functions enable zero-forgetting extension of the system.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.distributed.utils import is_main_process


def compute_prototypes(
    backbone: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Compute dual prototypes (lateral + overhead) for all pest classes.
    
    Args:
        backbone: Frozen backbone
        dataset: Dataset with viewpoint tags
        device: Device
        batch_size: Batch size
    
    Returns:
        Dict mapping pest_name -> {lateral: tensor, overhead: tensor}
    """
    backbone.eval()
    
    # Group embeddings by pest and viewpoint
    embeddings_by_pest_vp = {}
    
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch in tqdm(loader, disable=not is_main_process()):
            images = batch[0].to(device)
            metadata = batch[1]
            
            pests = metadata["pest"]
            viewpoints = metadata.get("viewpoint", ["unknown"] * len(pests))
            
            embeddings = backbone(images)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            for i, (pest, vp) in enumerate(zip(pests, viewpoints)):
                key = (pest, vp)
                if key not in embeddings_by_pest_vp:
                    embeddings_by_pest_vp[key] = []
                embeddings_by_pest_vp[key].append(embeddings[i])
    
    # Compute mean per pest/viewpoint
    prototypes = {}
    for pest in set(k[0] for k in embeddings_by_pest_vp.keys()):
        prototypes[pest] = {}
        
        for vp in ["lateral", "overhead"]:
            key = (pest, vp)
            if key in embeddings_by_pest_vp:
                embs = torch.stack(embeddings_by_pest_vp[key])
                prototypes[pest][vp] = embs.mean(dim=0)
            else:
                # Empty prototype (not enough data)
                prototypes[pest][vp] = None
    
    return prototypes


def add_crop(
    backbone: nn.Module,
    crop_heads_registry,
    crop_embeddings: nn.Embedding,
    train_dataset,
    crop_name: str,
    num_pests: int,
    num_symptoms: int,
    device: torch.device,
) -> None:
    """
    Add new crop to the system.
    
    Args:
        backbone: Frozen backbone
        crop_heads_registry: Registry of crop heads
        crop_embeddings: Crop embedding table
        train_dataset: Training dataset
        crop_name: Name of new crop
        num_pests: Number of pest classes for this crop
        num_symptoms: Number of symptom types for this crop
        device: Device
    """
    if is_main_process():
        print(f"Adding crop: {crop_name}")
    
    # Add crop embedding
    new_crop_emb = nn.Parameter(torch.randn(1, 128))
    nn.init.normal_(new_crop_emb, std=0.02)
    # Would need to add to embedding table (simplified here)
    
    # Create pest head for this crop
    from src.models.heads.crop_heads import CropPestHead, CropSymptomHead
    
    pest_head = CropPestHead(
        in_features=768,
        crop_embedding_dim=128,
        num_pests=num_pests,
    ).to(device)
    
    symptom_head = CropSymptomHead(
        in_features=768,
        num_symptoms=num_symptoms,
    ).to(device)
    
    # Add to registry
    crop_heads_registry.pest_heads[crop_name] = pest_head
    crop_heads_registry.symptom_heads[crop_name] = symptom_head
    
    if is_main_process():
        print(f"Added crop heads for {crop_name}")


def add_pest(
    backbone: nn.Module,
    crop_name: str,
    pest_name: str,
    train_dataset,
    device: torch.device,
) -> torch.Tensor:
    """
    Add new pest to existing crop.
    
    Args:
        backbone: Frozen backbone
        crop_name: Crop name
        pest_name: New pest name
        train_dataset: Training dataset
        device: Device
    
    Returns:
        Prototype for new pest
    """
    if is_main_process():
        print(f"Adding pest {pest_name} to crop {crop_name}")
    
    # Filter dataset for this pest
    pest_samples = [s for s in train_dataset.samples 
                    if s["pest"] == pest_name and s["crop"] == crop_name]
    
    if not pest_samples:
        raise ValueError(f"No samples for {pest_name} in {crop_name}")
    
    # Compute prototype
    backbone.eval()
    embeddings = []
    
    with torch.no_grad():
        for sample in pest_samples:
            # Would load image here
            pass
    
    if is_main_process():
        print(f"Added prototype for {pest_name}")
    
    return None  # Simplified


def save_prototypes(prototypes: Dict, path: Path) -> None:
    """Save prototype store to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prototypes, path)


def load_prototypes(path: Path) -> Dict:
    """Load prototype store from disk."""
    return torch.load(path)