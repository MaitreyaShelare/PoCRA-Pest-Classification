"""
Post-processing of predictions: formatting, mapping indices to names, confidence thresholding.
"""

from typing import Dict, List, Optional
import json


class PredictionFormatter:
    """Format predictions for different outputs."""
    
    def __init__(
        self,
        idx_to_pest: Dict[int, str],
        idx_to_crop: Dict[int, str],
        idx_to_type: Dict[int, str],
    ) -> None:
        """
        Initialize formatter.
        
        Args:
            idx_to_pest: Mapping index -> pest name
            idx_to_crop: Mapping index -> crop name
            idx_to_type: Mapping index -> image type
        """
        self.idx_to_pest = idx_to_pest
        self.idx_to_crop = idx_to_crop
        self.idx_to_type = idx_to_type
    
    def format_for_api(self, prediction: Dict) -> Dict:
        """
        Format prediction for API response.
        
        Args:
            prediction: Raw prediction dict
        
        Returns:
            API-formatted dict
        """
        return {
            "crop": prediction["crop"],
            "prediction": {
                "pest": prediction.get("pest", "UNKNOWN"),
                "confidence": prediction.get("pest_confidence", 0.0),
                "image_type": prediction["image_type"],
            },
            "metadata": {
                "ood_score": prediction.get("ood_score", 0.0),
                "is_ood": prediction.get("is_ood", False),
                "rejection_reason": prediction.get("rejection_reason", None),
            },
        }
    
    def format_for_farmer(self, prediction: Dict) -> str:
        """
        Format prediction as human-readable string for farmer.
        
        Args:
            prediction: Raw prediction dict
        
        Returns:
            Human-readable string
        """
        crop = prediction["crop"]
        
        if prediction.get("is_ood"):
            return f"[{crop}] Unknown pest detected. Please consult an expert."
        
        if prediction.get("rejection_reason"):
            return f"[{crop}] Low confidence prediction. Please retake photo."
        
        pest = prediction.get("pest", "Unknown")
        confidence = prediction.get("pest_confidence", 0.0)
        image_type = prediction["image_type"]
        
        if image_type == "healthy":
            return f"[{crop}] Crop appears HEALTHY ✓"
        elif image_type == "pest_body":
            return f"[{crop}] Detected: {pest} ({confidence:.1%} confidence)"
        elif image_type == "symptom":
            symptom = prediction.get("symptom", "Unknown symptom")
            return f"[{crop}] Detected symptom: {symptom} ({confidence:.1%} confidence)"
        else:
            return f"[{crop}] Unable to classify. Please retake photo."
    
    def format_for_report(self, predictions: List[Dict]) -> str:
        """
        Format batch predictions as report.
        
        Args:
            predictions: List of prediction dicts
        
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "PEST CLASSIFICATION REPORT",
            "=" * 60,
            f"Total images: {len(predictions)}\n",
        ]
        
        # Group by crop
        by_crop = {}
        for pred in predictions:
            crop = pred["crop"]
            if crop not in by_crop:
                by_crop[crop] = []
            by_crop[crop].append(pred)
        
        for crop, crop_preds in sorted(by_crop.items()):
            lines.append(f"\n{crop}")
            lines.append("-" * 40)
            
            healthy = sum(1 for p in crop_preds if p.get("pest") == "Healthy")
            ood = sum(1 for p in crop_preds if p.get("is_ood"))
            low_conf = sum(1 for p in crop_preds if p.get("rejection_reason"))
            classified = len(crop_preds) - healthy - ood - low_conf
            
            lines.append(f"  Healthy: {healthy}")
            lines.append(f"  Classified: {classified}")
            lines.append(f"  Low confidence: {low_conf}")
            lines.append(f"  Unknown (OOD): {ood}")
            
            # Top pests
            pests = {}
            for p in crop_preds:
                if not p.get("is_ood") and p.get("pest"):
                    pest = p["pest"]
                    pests[pest] = pests.get(pest, 0) + 1
            
            if pests:
                lines.append("\n  Top detected pests:")
                for pest, count in sorted(pests.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"    - {pest}: {count}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
    
    def to_json(self, predictions: List[Dict], pretty: bool = True) -> str:
        """
        Convert predictions to JSON.
        
        Args:
            predictions: List of predictions
            pretty: If True, pretty-print
        
        Returns:
            JSON string
        """
        formatted = [self.format_for_api(p) for p in predictions]
        return json.dumps(formatted, indent=2 if pretty else None)


class ConfidenceThresholder:
    """Apply confidence thresholding to predictions."""
    
    def __init__(self, threshold: float = 0.7) -> None:
        """
        Initialize thresholder.
        
        Args:
            threshold: Minimum confidence to accept prediction
        """
        self.threshold = threshold
    
    def apply(self, prediction: Dict) -> Dict:
        """
        Apply confidence threshold.
        
        Args:
            prediction: Prediction dict
        
        Returns:
            Prediction with rejection reason if below threshold
        """
        confidence = prediction.get("pest_confidence", 0.0)
        
        if confidence < self.threshold:
            prediction["rejected"] = True
            prediction["rejection_reason"] = f"Low confidence: {confidence:.3f} < {self.threshold}"
        else:
            prediction["rejected"] = False
        
        return prediction
    
    def apply_batch(self, predictions: List[Dict]) -> List[Dict]:
        """Apply to batch."""
        return [self.apply(p) for p in predictions]


def filter_rejected(predictions: List[Dict]) -> List[Dict]:
    """Filter out rejected predictions."""
    return [p for p in predictions if not p.get("rejected", False)]


def get_acceptance_rate(predictions: List[Dict]) -> float:
    """Get fraction of predictions that passed threshold."""
    total = len(predictions)
    accepted = sum(1 for p in predictions if not p.get("rejected", False))
    return accepted / total if total > 0 else 0.0


def get_ood_rate(predictions: List[Dict]) -> float:
    """Get fraction of predictions flagged as OOD."""
    total = len(predictions)
    ood = sum(1 for p in predictions if p.get("is_ood", False))
    return ood / total if total > 0 else 0.0