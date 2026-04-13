# Project Context

## Overview
This is a modular computer vision research framework built with PyTorch and DDP.

Pipeline:
configs → data → model → training → evaluation → experiments

---

## Folder Responsibilities

- src/data → dataset, dataloaders, transforms
- src/models → model definitions only (no training logic)
- src/train → training loop, optimization, scheduling
- src/eval → metrics and evaluation
- src/inference → prediction pipelines
- src/explainability → Grad-CAM and interpretability
- src/distributed → DDP setup and synchronization
- src/utils → shared utilities

---

## Rules

1. Models do NOT contain training logic
2. Training logic is only in src/train
3. All parameters come from configs/
4. Experiments must be saved in experiments/
5. Raw data must not be modified

---

## Entry Points

- Training → src/cli/train.py
- Evaluation → src/cli/evaluate.py
- Inference → src/cli/infer.py
- Explainability → src/cli/explain.py