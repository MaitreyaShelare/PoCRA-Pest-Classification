"""
Phase 1 trainer: ArcFace + SupCon for learning species-level embeddings.
This is the most important phase that trains the backbone.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig
from tqdm import tqdm

from src.train.trainer import BaseTrainer
from src.models.losses.arcface_loss import ArcFaceLoss
from src.models.losses.supcon_loss import SupConLoss
from src.core.registry import get_optimizer, get_scheduler
from src.distributed.sync import reduce_dict
from src.distributed.utils import is_main_process
from src.eval.metrics import macro_f1


class Phase1Trainer(BaseTrainer):
    """
    Phase 1 trainer: ArcFace + SupCon.
    
    Trains DINOv2 backbone to learn species-level pest embeddings.
    ArcFace provides the margin-based classification objective.
    SupCon provides the contrastive objective for tight clusters.
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
        Initialize Phase 1 trainer.
        
        Args:
            cfg: Configuration
            model: Model (backbone + ArcFace head)
            device: Device
            output_dir: Checkpoint directory
            rank: Process rank
            world_size: Total processes
        """
        super().__init__(cfg, model, device, output_dir, rank, world_size)
        
        # Get config sections
        self.cfg_phase = cfg.train.phase1
        
        # Create losses
        self.arcface_loss = ArcFaceLoss(
            in_features=cfg.model.embedding_dim,
            num_classes=cfg.num_pests,
            margin=cfg.model.arcface.margin,
            scale=cfg.model.arcface.scale,
        ).to(device)
        
        self.supcon_loss = SupConLoss(
            temperature=cfg.train.phase1.get("supcon_temperature", 0.07),
        )
        
        self.arcface_weight = self.cfg_phase.get("arcface_weight", 1.0)
        self.supcon_weight = self.cfg_phase.get("supcon_weight", 0.5)
        
        # Create optimizer (only for backbone + arcface head)
        self.optimizer = get_optimizer(self.cfg_phase, self.model)
        
        # Create scheduler
        self.scheduler = get_scheduler(self.cfg_phase, self.optimizer)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train one epoch with ArcFace + SupCon.
        
        Args:
            train_loader: Training DataLoader
        
        Returns:
            Dict of metrics
        """
        self.model.train()
        
        total_arcface_loss = 0.0
        total_supcon_loss = 0.0
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, disable=not is_main_process())
        
        for batch in pbar:
            images = batch[0].to(self.device)
            metadata = batch[1]
            
            # Get labels
            species_idx = metadata["species_idx"].to(self.device)
            pest_idx = metadata["pest_idx"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass through backbone
            embeddings = self.model(images)  # (B, 768)
            embeddings_norm = nn.functional.normalize(embeddings, p=2, dim=1)
            
            # ArcFace loss (classification)
            arcface_logits = self.arcface_loss.head(embeddings_norm, species_idx)
            arcface_loss = self.arcface_loss.loss_fn(arcface_logits, species_idx)
            
            # SupCon loss (contrastive)
            # Reshape for SupCon: treat batch as n_views=1
            supcon_loss = self.supcon_loss(
                embeddings_norm.unsqueeze(1),  # (B, 1, 768)
                labels=species_idx,
            )
            
            # Combined loss
            total_loss_val = (
                self.arcface_weight * arcface_loss +
                self.supcon_weight * supcon_loss
            )
            
            # Backward
            total_loss_val.backward()
            self.optimizer.step()
            
            # Metrics
            total_arcface_loss += arcface_loss.item() * images.size(0)
            total_supcon_loss += supcon_loss.item() * images.size(0)
            total_loss += total_loss_val.item() * images.size(0)
            total_samples += images.size(0)
            
            # Predictions
            all_preds.append(arcface_logits.argmax(dim=1).cpu())
            all_labels.append(species_idx.cpu())
            
            self.global_step += 1
            
            if is_main_process():
                pbar.set_description(
                    f"ArcFace: {arcface_loss.item():.4f}, "
                    f"SupCon: {supcon_loss.item():.4f}"
                )
        
        # Step scheduler
        self.scheduler.step()
        
        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        f1 = macro_f1(all_preds.numpy(), all_labels.numpy())
        
        metrics = {
            "arcface_loss": total_arcface_loss / total_samples,
            "supcon_loss": total_supcon_loss / total_samples,
            "loss": total_loss / total_samples,
            "f1": f1,
        }
        
        # Reduce across processes
        metrics = reduce_dict(metrics, average=True)
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            Dict of metrics
        """
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
                
                species_idx = metadata["species_idx"].to(self.device)
                
                # Forward
                embeddings = self.model(images)
                embeddings_norm = nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Loss
                arcface_logits = self.arcface_loss.head(embeddings_norm, species_idx)
                loss = self.arcface_loss.loss_fn(arcface_logits, species_idx)
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                # Predictions
                all_preds.append(arcface_logits.argmax(dim=1).cpu())
                all_labels.append(species_idx.cpu())
                
                pbar.set_description(f"Val loss: {loss.item():.4f}")
        
        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        f1 = macro_f1(all_preds.numpy(), all_labels.numpy())
        
        metrics = {
            "loss": total_loss / total_samples,
            "f1": f1,
        }
        
        metrics = reduce_dict(metrics, average=True)
        
        return metrics