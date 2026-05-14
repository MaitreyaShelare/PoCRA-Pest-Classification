"""
Phase 1 trainer: ArcFace + SupCon for learning species-level embeddings.
This is the most important phase that trains the backbone.
"""

import torch
import torch.nn as nn

from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
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
        """
        super().__init__(
            cfg,
            model,
            device,
            output_dir,
            rank,
            world_size,
        )

        # Config
        self.cfg_phase = cfg.train.phase1

        # Loss modules
        self.arcface_loss = ArcFaceLoss(
            in_features=cfg.model.dinov2.embedding_dim,
            num_classes=cfg.data.num_pests,
            margin=cfg.model.arcface.margin,
            scale=cfg.model.arcface.scale,
        ).to(device)

        self.supcon_loss = SupConLoss(
            temperature=cfg.train.phase1.get(
                "supcon_temperature",
                0.07,
            ),
        ).to(device)

        self.arcface_weight = self.cfg_phase.get(
            "arcface_weight",
            1.0,
        )

        self.supcon_weight = self.cfg_phase.get(
            "supcon_weight",
            0.5,
        )

        # Optimizer
        params = (
            list(self.model.parameters()) +
            list(self.arcface_loss.parameters())
        )

        self.optimizer = get_optimizer(
            self.cfg_phase,
            params,
        )

        # Scheduler
        self.scheduler = get_scheduler(
            self.cfg_phase,
            self.optimizer,
        )

        # AMP scaler
        self.scaler = GradScaler("cuda")

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train one epoch.
        """

        self.model.train()

        if hasattr(self.arcface_loss, "train"):
            self.arcface_loss.train()

        total_arcface_loss = 0.0
        total_supcon_loss = 0.0
        total_loss = 0.0
        total_samples = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(
            train_loader,
            disable=not is_main_process(),
        )

        for batch in pbar:

            images = batch[0].to(
                self.device,
                non_blocking=True,
            )

            metadata = batch[1]

            species_idx = metadata["species_idx"].to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(
                set_to_none=True,
            )

            # with autocast("cuda"):
            with autocast(device_type='cuda'):

                # if is_main_process():
                #     print("before forward", flush=True)

                # Forward
                embeddings = self.model(images)

                # if is_main_process():
                #     print("after forward", flush=True)

                embeddings_norm = nn.functional.normalize(
                    embeddings,
                    p=2,
                    dim=1,
                )

                # ArcFace
                arcface_logits = self.arcface_loss.head(
                    embeddings_norm,
                    species_idx,
                )

                arcface_loss = self.arcface_loss.loss_fn(
                    arcface_logits,
                    species_idx,
                )

                # SupCon in FP32 for numerical stability
                supcon_loss = self.supcon_loss(
                    embeddings_norm.float(),
                    labels=species_idx,
                )

                # Combined loss
                total_loss_val = (
                    self.arcface_weight * arcface_loss +
                    self.supcon_weight * supcon_loss
                )

            # Backward
            self.scaler.scale(
                total_loss_val
            ).backward()

            # if is_main_process():
            #     print("after backward", flush=True)

            self.scaler.step(
                self.optimizer
            )

            # if is_main_process():
            #     print("after step", flush=True)

            self.scaler.update()

            # Metrics
            batch_size = images.size(0)

            total_arcface_loss += (
                arcface_loss.item() * batch_size
            )

            total_supcon_loss += (
                supcon_loss.item() * batch_size
            )

            total_loss += (
                total_loss_val.item() * batch_size
            )

            total_samples += batch_size

            all_preds.append(
                arcface_logits.argmax(dim=1).detach().cpu()
            )

            all_labels.append(
                species_idx.detach().cpu()
            )

            self.global_step += 1

            if is_main_process():

                pbar.set_description(
                    f"ArcFace: {arcface_loss.item():.4f}, "
                    f"SupCon: {supcon_loss.item():.4f}"
                )

        self.scheduler.step()

        # Edge case
        if total_samples == 0:

            return {
                "arcface_loss": 0.0,
                "supcon_loss": 0.0,
                "loss": 0.0,
                "f1": 0.0,
            }

        # Metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        f1 = macro_f1(
            all_preds.numpy(),
            all_labels.numpy(),
        )

        metrics = {
            "arcface_loss":
                total_arcface_loss / total_samples,

            "supcon_loss":
                total_supcon_loss / total_samples,

            "loss":
                total_loss / total_samples,

            "f1":
                f1,
        }

        if self.world_size > 1:

            reduced = reduce_dict(
                {
                    "arcface_loss": metrics["arcface_loss"],
                    "supcon_loss": metrics["supcon_loss"],
                    "loss": metrics["loss"],
                },
                average=True,
            )

            metrics.update(reduced)

        return metrics

    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate model.
        """

        self.model.eval()

        if hasattr(self.arcface_loss, "eval"):
            self.arcface_loss.eval()

        total_loss = 0.0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            pbar = tqdm(
                val_loader,
                disable=not is_main_process(),
            )

            for batch in pbar:

                images = batch[0].to(
                    self.device,
                    non_blocking=True,
                )

                metadata = batch[1]

                species_idx = metadata["species_idx"].to(
                    self.device,
                    non_blocking=True,
                )

                # with autocast():
                with autocast(device_type='cuda'):

                    # Forward
                    embeddings = self.model(images)

                    embeddings_norm = nn.functional.normalize(
                        embeddings,
                        p=2,
                        dim=1,
                    )

                    # ArcFace
                    arcface_logits = self.arcface_loss.head(
                        embeddings_norm,
                        species_idx,
                    )

                    loss = self.arcface_loss.loss_fn(
                        arcface_logits,
                        species_idx,
                    )

                batch_size = images.size(0)

                total_loss += (
                    loss.item() * batch_size
                )

                total_samples += batch_size

                all_preds.append(
                    arcface_logits.argmax(dim=1)
                    .detach()
                    .cpu()
                )

                all_labels.append(
                    species_idx.detach().cpu()
                )

                if is_main_process():

                    pbar.set_description(
                        f"Val loss: {loss.item():.4f}"
                    )

        # Edge case
        if total_samples == 0:

            return {
                "loss": 0.0,
                "f1": 0.0,
            }
        

        # Metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        f1 = macro_f1(
            all_preds.numpy(),
            all_labels.numpy(),
        )

        metrics = {
            "loss":
                total_loss / total_samples,

            "f1":
                f1,
        }


        return metrics
