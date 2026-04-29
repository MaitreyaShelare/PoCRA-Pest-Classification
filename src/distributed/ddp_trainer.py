"""
DDP Trainer wrapper: handles model wrapping, checkpoint sync, etc.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from src.distributed.setup import is_initialized, get_rank, is_main_process


class DDPTrainer:
    """
    Wrapper trainer that handles DDP model wrapping and synchronization.
    
    Usage:
        trainer = DDPTrainer(cfg, model, device, output_dir)
        # Automatically wrapped for DDP if world_size > 1
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        device: torch.device,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize DDP trainer.
        
        Args:
            cfg: Configuration
            model: Model to wrap
            device: Device (cuda:rank)
            output_dir: Where to save checkpoints (rank 0 only)
        """
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.rank = get_rank()
        self.world_size = 1 if not is_initialized() else torch.distributed.get_world_size()
        
        # Wrap model for DDP if distributed
        if is_initialized() and self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                model.to(device),
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )
        else:
            self.model = model.to(device)
        
        if self.rank == 0:
            print(
                f"Model initialized on device {device}\n"
                f"  Distributed: {self.world_size > 1}\n"
                f"  World size: {self.world_size}"
            )
    
    def save_checkpoint(self, path: Path, epoch: int, step: int) -> None:
        """
        Save model checkpoint (rank 0 only).
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            step: Current global step
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
            "step": step,
            "model_state_dict": state_dict,
            "config": self.cfg,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """
        Load model checkpoint (all ranks).
        
        Args:
            path: Path to checkpoint
        
        Returns:
            Checkpoint dict (epoch, step, etc.)
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state (unwrap if DDP)
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if is_main_process():
            print(f"Checkpoint loaded: {path}")
            print(f"  Epoch: {checkpoint.get('epoch', 0)}")
            print(f"  Step: {checkpoint.get('step', 0)}")
        
        return checkpoint
    
    def synchronize(self) -> None:
        """
        Synchronize all processes (barrier).
        Useful before saving checkpoints or printing.
        """
        if is_initialized():
            torch.distributed.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """
        Reduce tensor across all processes.
        
        Args:
            tensor: Tensor to reduce
            op: "mean", "sum", "max", "min"
        
        Returns:
            Reduced tensor
        """
        if not is_initialized():
            return tensor
        
        tensor = tensor.to(self.device)
        
        if op == "mean":
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            tensor /= self.world_size
        elif op == "sum":
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        elif op == "max":
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        elif op == "min":
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        
        return tensor