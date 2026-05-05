"""Loss functions."""

from .arcface_loss import ArcFaceHead, ArcFaceLoss, ArcFaceWithSupConLoss
from .supcon_loss import SupConLoss

__all__ = [
    "ArcFaceHead",
    "ArcFaceLoss",
    "ArcFaceWithSupConLoss",
    "SupConLoss",
]