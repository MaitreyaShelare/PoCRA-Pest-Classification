"""
Batch prediction with caching and result aggregation.
For processing large numbers of images efficiently.
"""

import torch
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import DataLoader, Dataset
import csv
from tqdm import tqdm

from src.data.transforms import get_transforms
from src.inference.postprocess import PredictionFormatter, ConfidenceThresholder


class BatchPredictor:
    """Efficient batch predictor with result aggregation."""
    
    def __init__(
        self,
        predictor,
        batch_size: int = 32,
        num_workers: int = 4,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        Initialize batch predictor.
        
        Args:
            predictor: PestClassificationPredictor instance
            batch_size: Batch size
            num_workers: Number of DataLoader workers
            confidence_threshold: Confidence threshold
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.thresholder = ConfidenceThresholder(confidence_threshold)
    
    def predict_from_csv(
        self,
        csv_path: Path,
        image_dir: Path,
        output_path: Path,
    ) -> List[Dict]:
        """
        Predict on images listed in CSV.
        
        Args:
            csv_path: Path to CSV with image paths
            image_dir: Root directory for images
            output_path: Path to save results
        
        Returns:
            List of predictions
        """
        # Load file paths from CSV
        filepaths = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                filepaths.append(row["filepath"])
        
        # Predict
        results = self.predict_from_paths(
            filepaths,
            image_dir,
            output_path,
        )
        
        return results
    
    def predict_from_paths(
        self,
        filepaths: List[str],
        image_dir: Path,
        output_path: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Predict on images from file paths.
        
        Args:
            filepaths: List of image file paths (relative to image_dir)
            image_dir: Root directory
            output_path: Path to save results CSV
        
        Returns:
            List of predictions
        """
        from PIL import Image
        
        class PathDataset(Dataset):
            def __init__(self, filepaths: List[str], image_dir: Path):
                self.filepaths = filepaths
                self.image_dir = image_dir
                self.transform = get_transforms("val")
            
            def __len__(self):
                return len(self.filepaths)
            
            def __getitem__(self, idx):
                filepath = self.filepaths[idx]
                img = Image.open(self.image_dir / filepath).convert("RGB")
                return self.transform(img), filepath
        
        dataset = PathDataset(filepaths, image_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        
        all_results = []
        
        with torch.no_grad():
            for batch_imgs, batch_fps in tqdm(loader, desc="Predicting"):
                # Extract crop from filepath (assumes structure: crop/...)
                crop_names = [Path(fp).parts[0] for fp in batch_fps]
                
                batch_results = self.predictor.predict_batch(batch_imgs, crop_names)
                
                # Apply thresholding
                batch_results = self.thresholder.apply_batch(batch_results)
                
                # Attach filepath
                for result, filepath in zip(batch_results, batch_fps):
                    result["filepath"] = filepath
                    all_results.append(result)
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", newline="") as f:
                fieldnames = list(all_results[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
        
        return all_results
    
    def get_summary(self, results: List[Dict]) -> Dict:
        """
        Get summary statistics from batch results.
        
        Args:
            results: List of predictions
        
        Returns:
            Summary dict
        """
        total = len(results)
        accepted = sum(1 for r in results if not r.get("rejected"))
        ood = sum(1 for r in results if r.get("is_ood"))
        
        # By crop
        by_crop = {}
        for r in results:
            crop = r["crop"]
            if crop not in by_crop:
                by_crop[crop] = {"total": 0, "accepted": 0}
            by_crop[crop]["total"] += 1
            if not r.get("rejected"):
                by_crop[crop]["accepted"] += 1
        
        return {
            "total_images": total,
            "accepted": accepted,
            "acceptance_rate": accepted / total if total > 0 else 0,
            "ood_count": ood,
            "ood_rate": ood / total if total > 0 else 0,
            "by_crop": by_crop,
        }