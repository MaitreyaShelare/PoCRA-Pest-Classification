"""
Explainability: Grad-CAM, attention maps, saliency.
"""

import argparse
from pathlib import Path
import torch
from PIL import Image

from src.core.config import load_config
from src.data.transforms import get_transforms
from src.explainability.gradcam import GradCAM
from src.explainability.saliency import compute_saliency


def main():
    parser = argparse.ArgumentParser(description="Generate explainability maps")
    parser.add_argument("--config-dir", type=Path, default="configs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default="outputs/explain")
    parser.add_argument("--method", choices=["gradcam", "saliency"], default="gradcam")
    args = parser.parse_args()

    cfg = load_config(args.config_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    transform = get_transforms("val")
    img = Image.open(args.image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Load model
    # (Load checkpoint, build model...)

    # Generate explanation
    if args.method == "gradcam":
        explainer = GradCAM(model, target_layer="layer4")
        cam = explainer(img_tensor)
    elif args.method == "saliency":
        cam = compute_saliency(img_tensor)

    # Save
    output_path = args.output_dir / f"{args.image.stem}_{args.method}.png"
    # (Save visualization...)

    print(f"Explanation saved to {output_path}")


if __name__ == "__main__":
    main()