import os
import sys
sys.path.append(os.path.abspath("src"))

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from core.seed import set_seed
from core.logger import get_logger
from core.config import save_config

from distributed.setup import setup_ddp, cleanup_ddp
from distributed.utils import is_main_process

from utils.experiment import create_experiment_dir
from utils.metrics_logger import save_metrics

# ---- Dummy placeholders (replace later) ----
from models.classifier import ImageClassifier


def main():
    # -------------------------
    # Setup DDP
    # -------------------------
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # -------------------------
    # Config (replace with YAML later)
    # -------------------------
    cfg = {
        "seed": 42,
        "epochs": 5,
        "lr": 1e-3,
        "model": "resnet18",
        "num_classes": 10,
        "batch_size": 32
    }

    set_seed(cfg["seed"])

    # -------------------------
    # Experiment setup
    # -------------------------
    if is_main_process():
        exp_dir = create_experiment_dir(exp_name=cfg["model"])
        save_config(cfg, os.path.join(exp_dir, "config.yaml"))
    else:
        exp_dir = None

    # Broadcast exp_dir to all processes
    if torch.distributed.is_initialized():
        exp_dir = [exp_dir]
        torch.distributed.broadcast_object_list(exp_dir, src=0)
        exp_dir = exp_dir[0]

    logger = get_logger(os.path.join(exp_dir, "logs"), rank=local_rank)

    # -------------------------
    # Model
    # -------------------------
    model = ImageClassifier(cfg["model"], cfg["num_classes"]).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    # -------------------------
    # Dummy data (replace later)
    # -------------------------
    data = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, cfg["num_classes"], (100,))

    dataset = list(zip(data, labels))
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"])

    # -------------------------
    # Training loop
    # -------------------------
    metrics = {}

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0

        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if is_main_process():
            logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        metrics[f"epoch_{epoch}"] = {"loss": avg_loss}

        # Save checkpoint
        if is_main_process():
            torch.save(
                model.module.state_dict(),
                os.path.join(exp_dir, "checkpoints", f"epoch_{epoch}.pth")
            )

    # -------------------------
    # Save metrics
    # -------------------------
    if is_main_process():
        save_metrics(metrics, os.path.join(exp_dir, "metrics.json"))

    cleanup_ddp()


if __name__ == "__main__":
    main()