"""
Augmentation pipelines for training and validation.
Separate configs for each phase (train/val/test).
"""

import torchvision.transforms as transforms
from torchvision.transforms import RandAugment


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Get augmentation pipeline.
    
    Args:
        split: "train", "val", or "test"
    
    Returns:
        Composed transforms
    """
    
    if split == "train":
        return transforms.Compose([
            # Geometric
            transforms.RandomResizedCrop(
                224,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=0.5,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            
            # Color
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.RandomEqualize(p=0.3),
            
            # Auto-augment
            RandAugment(
                num_ops=2,
                magnitude=9,
                num_magnitude_bins=31,
            ),
            
            # Normalization
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def get_train_transforms() -> transforms.Compose:
    """Alias for get_transforms("train")."""
    return get_transforms("train")


def get_val_transforms() -> transforms.Compose:
    """Alias for get_transforms("val")."""
    return get_transforms("val")


def get_test_transforms() -> transforms.Compose:
    """Alias for get_transforms("test")."""
    return get_transforms("test")