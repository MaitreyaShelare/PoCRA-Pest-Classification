"""
PestDataset: Load images and labels from cleaned dataset.
Ground truth comes from labels.csv, not folder structure.
"""

from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import Tuple, Dict, Optional
import csv
from PIL import Image
import torchvision.transforms as transforms


class PestDataset(Dataset):
    """
    Load pest classification dataset from directory + labels.csv.
    
    Labels CSV format:
        filepath,crop,pest,image_type,species_id,split
    """
    
    def __init__(
        self,
        labels_csv: Path,
        data_root: Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        exclude_ambiguous: bool = True,
    ) -> None:
        """
        Initialize dataset.
        
        Args:
            labels_csv: Path to labels.csv
            data_root: Root directory of images
            split: "train", "val", or "test"
            transform: Torchvision transform pipeline
            exclude_ambiguous: Skip rows with image_type="ambiguous"
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.split = split
        self.exclude_ambiguous = exclude_ambiguous
        self.samples = []

        # Load labels.csv
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by split
                if row.get("split") != split:
                    continue
                
                # Exclude ambiguous if requested
                if exclude_ambiguous and row.get("image_type") == "ambiguous":
                    continue
                
                # Verify file exists
                img_path = self.data_root / row["filepath"]
                if not img_path.exists():
                    print(f"WARNING: Image not found: {img_path}")
                    continue
                
                self.samples.append(row)

        # Build indices for all categorical columns
        self.crops = sorted(set(s["crop"] for s in self.samples))
        self.pests = sorted(set(s["pest"] for s in self.samples))
        self.image_types = sorted(set(s["image_type"] for s in self.samples))
        self.species = sorted(set(s["species_id"] for s in self.samples if s["species_id"]))

        # Mappings: name -> index
        self.crop_to_idx = {c: i for i, c in enumerate(self.crops)}
        self.pest_to_idx = {p: i for i, p in enumerate(self.pests)}
        self.type_to_idx = {t: i for i, t in enumerate(self.image_types)}
        self.species_to_idx = {s: i for i, s in enumerate(self.species)}

        # Reverse mappings: index -> name
        self.idx_to_crop = {i: c for c, i in self.crop_to_idx.items()}
        self.idx_to_pest = {i: p for p, i in self.pest_to_idx.items()}
        self.idx_to_type = {i: t for t, i in self.type_to_idx.items()}

        print(
            f"Loaded {len(self.samples)} samples ({split})\n"
            f"  Crops: {len(self.crops)}, Pests: {len(self.pests)}, "
            f"Types: {len(self.image_types)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get one sample.
        
        Returns:
            (image_tensor, metadata_dict)
        """
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_root / sample["filepath"]
        img = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Build metadata dict
        metadata = {
            "filepath": sample["filepath"],
            "crop": sample["crop"],
            "crop_idx": self.crop_to_idx[sample["crop"]],
            "pest": sample["pest"],
            "pest_idx": self.pest_to_idx[sample["pest"]],
            "image_type": sample["image_type"],
            "type_idx": self.type_to_idx[sample["image_type"]],
            "species_id": sample.get("species_id", "unknown"),
            "species_idx": self.species_to_idx.get(
                sample.get("species_id"), -1
            ),  # -1 for Healthy
        }
        
        return img, metadata

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss.
        For handling imbalanced classes.
        
        Returns:
            Tensor of shape (num_pests,) with class weights
        """
        from collections import Counter
        
        pest_counts = Counter(s["pest"] for s in self.samples)
        total = len(self.samples)
        
        weights = []
        for pest in self.pests:
            weight = total / (pest_counts[pest] * len(self.pests))
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)

    def get_crop_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency weights per crop.
        
        Returns:
            Tensor of shape (num_crops,) with weights
        """
        from collections import Counter
        
        crop_counts = Counter(s["crop"] for s in self.samples)
        total = len(self.samples)
        
        weights = []
        for crop in self.crops:
            weight = total / (crop_counts[crop] * len(self.crops))
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)

    def get_species_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency weights per species (for Stage 2).
        
        Returns:
            Tensor of shape (num_species,) with weights
        """
        from collections import Counter
        
        species_counts = Counter(
            s["species_id"] for s in self.samples if s["species_id"] != "None"
        )
        total = len([s for s in self.samples if s["species_id"] != "None"])
        
        weights = []
        for species in self.species:
            weight = total / (species_counts[species] * len(self.species))
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_counts(self) -> Dict[str, int]:
        """Get count statistics."""
        from collections import Counter
        
        return {
            "total": len(self.samples),
            "crops": len(self.crops),
            "pests": len(self.pests),
            "healthy": sum(1 for s in self.samples if s["pest"] == "Healthy"),
            "pest_body": sum(1 for s in self.samples if s["image_type"] == "pest_body"),
            "symptom": sum(1 for s in self.samples if s["image_type"] == "symptom"),
        }