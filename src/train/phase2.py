"""
Phase 2 trainer: Train image type router (pest_body / symptom / healthy).
Backbone is frozen from Phase 1.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from omegaconf import DictConfig
from tqdm import tqdm

from src.train.trainer import BaseTrainer
from src.core.registry import get_optimizer, get_scheduler
from src.distributed.sync import reduce_dict
from src.distributed.utils import is_main_process
from src.eval.metrics import macro_f1


class Phase2Trainer(BaseTrainer):
    """
    Phase 2 trainer: Image type router.
    
    Trains linear head to classify image_type: pest_body, symptom, healthy.
    Backbone frozen from Phase 1.
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
        Initialize Phase 2 trainer.
        
        Args:
            cfg: Configuration
            model: Model (backbone frozen + router head)
            device: Device
            output_dir: Checkpoint directory
            rank: Process rank
            world_size: Total processes
        """
        super().__init__(cfg, model, device, output_dir, rank, world_size)
        
        self.cfg_phase = cfg.train.phase2
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Optimizer (only head, backbone frozen)
        self.optimizer = get_optimizer(self.cfg_phase, self.model)
        self.scheduler = get_scheduler(self.cfg_phase, self.optimizer)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, disable=not is_main_process())
        
        for batch in pbar:
            images = batch[0].to(self.device)
            metadata = batch[1]
            
            type_idx = metadata["type_idx"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            logits = self.model(images)  # Router head output
            loss = self.loss_fn(logits, type_idx)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(type_idx.cpu())
            
            self.global_step += 1
            
            if is_main_process():
                pbar.set_description(f"Loss: {loss.item():.4f}")
        
        self.scheduler.step()
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1 = macro_f1(all_preds.numpy(), all_labels.numpy())
        
        metrics = {
            "loss": total_loss / total_samples,
            "f1": f1,
        }
        
        metrics = reduce_dict(metrics, average=True)
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, disable=not is_main_process())
            
            for batch in pbar:
                images = batch[0].to(self.device)
                metadata = batch[1]
                type_idx = metadata["type_idx"].to(self.device)
                
                logits = self.model(images)
                loss = self.loss_fn(logits, type_idx)
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(type_idx.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1 = macro_f1(all_preds.numpy(), all_labels.numpy())
        
        metrics = {
            "loss": total_loss / total_samples,
            "f1": f1,
        }
        
        metrics = reduce_dict(metrics, average=True)
        return metrics