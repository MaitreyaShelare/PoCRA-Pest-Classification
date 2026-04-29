"""
Experiment logging to file and Weights & Biases.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ExperimentLogger:
    """Log metrics to file and optionally to Weights & Biases."""
    
    def __init__(
        self,
        exp_dir: Path,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        rank: int = 0,
    ) -> None:
        """
        Initialize logger.
        
        Args:
            exp_dir: Experiment directory for log file
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            rank: Process rank (only rank 0 logs)
        """
        self.exp_dir = Path(exp_dir)
        self.rank = rank
        self.use_wandb = use_wandb and rank == 0
        
        # File logger (rank 0 only)
        if rank == 0:
            self.log_path = self.exp_dir / "metrics.log"
            self.log_file = open(self.log_path, "w")
            self.log_file.write(f"Experiment started at {datetime.now().isoformat()}\n")
            self.log_file.write("-" * 80 + "\n")
        
        # W&B logger
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, dir=str(self.exp_dir))
                self.wandb = wandb
            except ImportError:
                self.use_wandb = False

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics to file and W&B.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Global step number
        """
        if self.rank != 0:
            return
        
        # File logging
        line = f"Step {step:6d} | "
        line += " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.log_file.write(line + "\n")
        self.log_file.flush()
        
        # W&B logging
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def log_message(self, message: str) -> None:
        """
        Log a message to file.
        
        Args:
            message: Message to log
        """
        if self.rank != 0:
            return
        
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def close(self) -> None:
        """Close log file and finish W&B run."""
        if self.rank != 0:
            return
        
        if hasattr(self, "log_file"):
            self.log_file.close()
        
        if self.use_wandb:
            self.wandb.finish()