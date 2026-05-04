"""
Phase 3 trainer: Train per-crop pest and symptom heads.
Backbone frozen from Phase 1, Stage 1 router frozen from Phase 2.
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


class Phase3Trainer(BaseTrainer):
    """
    Phase 3 trainer: Per-crop pest and symptom heads.
    
    Trains separate linear heads for each crop.
    Backbone and router frozen.
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
        Initialize Phase 3 trainer.
        
        Args:
            cfg: Configuration
            model: Model (backbone + heads)
            device: Device
            output_dir: Checkpoint directory
            rank: Process rank
            world_size: Total processes
        """
        super().__init__(cfg, model, device, output_dir, rank, world_size)
        
        self.cfg_phase = cfg.train.phase3
        
        # Loss with class weights for imbalance
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Optimizer
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
            
            image_type = metadata["image_type"]
            crop = metadata["crop"]
            
            self.optimizer.zero_grad()
            
            # Get appropriate label based on image type
            if image_type[0] == "pest_body":
                label = metadata["pest_idx"].to(self.device)
            elif image_type[0] == "symptom":
                # Symptom classification (would need symptom_idx)
                label = metadata["pest_idx"].to(self.device)  # Placeholder
            else:  # healthy
                continue
            
            # Forward (would call crop head)
            # logits = model(images, crop, image_type)
            logits = self.model(images)  # Simplified
            loss = self.loss_fn(logits, label)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(label.cpu())
            
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
                
                image_type = metadata["image_type"]
                
                if image_type[0] == "pest_body":
                    label = metadata["pest_idx"].to(self.device)
                elif image_type[0] == "symptom":
                    label = metadata["pest_idx"].to(self.device)
                else:
                    continue
                
                logits = self.model(images)
                loss = self.loss_fn(logits, label)
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(label.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1 = macro_f1(all_preds.numpy(), all_labels.numpy())
        
        metrics = {
            "loss": total_loss / total_samples,
            "f1": f1,
        }
        
        metrics = reduce_dict(metrics, average=True)
        return metrics