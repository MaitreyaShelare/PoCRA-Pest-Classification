"""Head modules."""

from .arcface_head import ArcFaceHead
from .router_head import RouterHead
from .crop_heads import CropHeadsRegistry

__all__ = ["ArcFaceHead", "RouterHead", "CropHeadsRegistry"]