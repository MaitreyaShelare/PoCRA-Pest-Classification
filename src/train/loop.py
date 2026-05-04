"""
Training and validation loop functions.
Reusable across all phases.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Callable, Optional
from tqdm import tqdm

from src.distributed.sync import reduce_dict
from src.distributed.utils import is_main_process


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 50,
) -> Dict[str, float]:
    """
    Train one epoch.
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device
        scaler: AMP GradScaler (optional)
        log_interval: Print loss every N batches
    
    Returns:
        Dict of metrics
    """
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, disable=not is_main_process())
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        images = batch[0].to(device)
        labels = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in batch[1].items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = loss_fn(output, labels)
        else:
            output = model(images)
            loss = loss_fn(output, labels)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Metrics
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        if (batch_idx + 1) % log_interval == 0 and is_main_process():
            avg_loss = total_loss / total_samples
            pbar.set_description(f"Loss: {avg_loss:.4f}")
    
    return {
        "loss": total_loss / total_samples,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    metric_fn: Callable,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate on validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation DataLoader
        metric_fn: Function to compute metrics
        device: Device
    
    Returns:
        Dict of metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, disable=not is_main_process())
        
        for batch in pbar:
            images = batch[0].to(device)
            labels_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch[1].items()}
            
            # Forward pass
            output = model(images)
            
            # Collect for metric computation
            all_preds.append(output.cpu())
            all_labels.append(labels_dict)
            total_samples += images.size(0)
            
            pbar.set_description("Validating")
    
    # Compute metrics
    metrics = metric_fn(all_preds, all_labels)
    
    # Reduce across all processes
    metrics = reduce_dict(metrics, average=True)
    
    return metrics