"""
Base trainer class with common training loop logic.
Subclassed by Phase1Trainer, Phase2Trainer, Phase3Trainer.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List
from omegaconf import DictConfig
from abc import ABC, abstractmethod

from src.core.logger import ExperimentLogger
from src.distributed.utils import is_main_process, barrier


class BaseTrainer(ABC):
    """
    Abstract base trainer.
    Defines interface and common methods for all trainers.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        device: torch.device,
        output_dir: Optional[Path] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """
        Initialize base trainer.
        
        Args:
            cfg: Configuration
            model: Model to train
            device: Device (cuda:rank)
            output_dir: Directory to save checkpoints (rank 0 only)
            rank: Process rank
            world_size: Total processes
        """
        self.cfg = cfg
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.rank = rank
        self.world_size = world_size
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = -float("inf")
        self.epochs_no_improve = 0
        
        # Logger (rank 0 only)
        if self.output_dir and is_main_process():
            self.logger = ExperimentLogger(
                exp_dir=self.output_dir,
                use_wandb=cfg.get("use_wandb", False),
                rank=rank,
            )
        else:
            self.logger = None
    
    @abstractmethod
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train one epoch.
        
        Args:
            train_loader: Training DataLoader
        
        Returns:
            Dict of metrics
        """
        pass
    
    @abstractmethod
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            Dict of metrics
        """
        pass
    
    def should_stop(self, current_metric: float, patience: int) -> bool:
        """
        Check if early stopping condition is met.
        
        Args:
            current_metric: Current validation metric (higher is better)
            patience: Number of epochs with no improvement before stopping
        
        Returns:
            True if should stop
        """
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            return self.epochs_no_improve >= patience
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict = None,
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Optional metrics dict to save
        """
        if not is_main_process():
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get model state (unwrap if DDP)
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": state_dict,
            "best_metric": self.best_metric,
            "config": self.cfg,
        }
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, path)
        
        if self.logger:
            self.logger.log_message(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state (unwrap if DDP)
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", -float("inf"))
        
        if is_main_process():
            if self.logger:
                self.logger.log_message(f"Checkpoint loaded: {path}")
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Log metrics to file and W&B.
        
        Args:
            metrics: Dict of metric_name -> value
            epoch: Current epoch
        """
        if self.logger:
            self.logger.log_metrics(metrics, step=epoch)
    
    def close(self) -> None:
        """Cleanup: close logger, etc."""
        if self.logger:
            self.logger.close()