"""PoCRA Pest Classification System."""

__version__ = "1.0.0"
__author__ = "Agricultural AI Lab"

from . import core
from . import data
from . import models
from . import train
from . import eval
from . import inference
from . import distributed

__all__ = [
    "core",
    "data", 
    "models",
    "train",
    "eval",
    "inference",
    "distributed",
]