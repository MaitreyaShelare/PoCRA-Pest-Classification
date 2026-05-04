"""Loss functions."""

from .arcface_loss import ArcFaceHead, ArcFaceLoss
from .supcon_loss import SupConLoss

__all__ = ["ArcFaceHead", "ArcFaceLoss", "SupConLoss"]