
---

# Standard Operating Procedure (SOP)

## Computer Vision Research Template (PyTorch + DDP)

---

## 1. Purpose

This repository provides a **modular, reproducible, and scalable framework** for conducting machine learning experiments in computer vision. It is designed to support:

* Rapid experimentation
* Multi-GPU training (DDP)
* Clear separation of concerns
* Reproducible research workflows
* Easy extension to new datasets, models, and tasks

---

## 2. Design Philosophy

The repository follows these principles:

1. **Config-driven experimentation**
   All experiments should be controlled via configuration files (`configs/`), not hardcoded values.

2. **Separation of concerns**
   Each module handles a single responsibility (data, model, training, evaluation, etc.).

3. **Reproducibility first**
   Every experiment must be reproducible using saved configs, seeds, and checkpoints.

4. **Modularity and extensibility**
   Components (models, datasets, losses) can be swapped independently.

5. **Experiment traceability**
   Every run is logged and stored in a structured format under `experiments/`.

---

## 3. High-Level Workflow

```
configs → data → model → training → evaluation → logging → experiments
```

Execution starts from CLI scripts and flows through modular components.

---

## 4. Directory Structure Overview

### 4.1 Root Directory

| Path               | Description                                       |
| ------------------ | ------------------------------------------------- |
| `README.md`        | Project overview and usage instructions           |
| `requirements.txt` | Python dependencies                               |
| `pyproject.toml`   | Build/system configuration                        |
| `Makefile`         | Command shortcuts for training and evaluation     |
| `notes.md`         | Research notes, observations, and experiment logs |
| `tests/`           | Unit tests for core components                    |
| `docs/`            | Documentation (including this SOP)                |

---

## 5. Configuration System (`configs/`)

This directory defines all experiment parameters.

| Path           | Purpose                                 |
| -------------- | --------------------------------------- |
| `default.yaml` | Global defaults (seed, device, logging) |
| `data/`        | Dataset configurations                  |
| `model/`       | Model configurations                    |
| `train/`       | Training configurations                 |

### Key Rule:

> Do not hardcode experiment parameters in code. Always use configs.

---

## 6. Data Management (`data/`)

| Folder       | Purpose                                 |
| ------------ | --------------------------------------- |
| `raw/`       | Original dataset (never modified)       |
| `processed/` | Cleaned/transformed dataset             |
| `splits/`    | Train/validation/test splits            |
| `metadata/`  | Class mappings, statistics, annotations |

---

## 7. Experiment Tracking (`experiments/`)

Each experiment generates a dedicated folder:

```
experiments/exp_name_timestamp/
├── config.yaml
├── logs/
├── checkpoints/
├── metrics.json
├── predictions/
├── gradcam/
```

### Purpose:

* Track all runs
* Enable reproducibility
* Support comparisons and ablations

---

## 8. Output Storage (`outputs/`)

Stores aggregated or final outputs:

* plots
* reports
* evaluation summaries

---

## 9. Model Artifacts (`models/`)

| Folder        | Purpose                          |
| ------------- | -------------------------------- |
| `pretrained/` | External pretrained weights      |
| `exported/`   | Saved models (TorchScript, ONNX) |

---

## 10. Scripts (`scripts/`)

| Script            | Purpose                           |
| ----------------- | --------------------------------- |
| `run_ddp.sh`      | Multi-GPU training using torchrun |
| `run_train.sh`    | Single-GPU training               |
| `prepare_data.py` | Dataset preprocessing             |

---

## 11. Source Code (`src/`)

This is the core codebase.

---

### 11.1 CLI (`src/cli/`)

Entry points for execution.

| File          | Purpose                |
| ------------- | ---------------------- |
| `train.py`    | Training pipeline      |
| `evaluate.py` | Model evaluation       |
| `infer.py`    | Inference              |
| `explain.py`  | Model interpretability |

---

### 11.2 Core (`src/core/`)

Infrastructure utilities.

| File          | Purpose                              |
| ------------- | ------------------------------------ |
| `config.py`   | Load/save configuration              |
| `seed.py`     | Set random seeds                     |
| `device.py`   | Device management (CPU/GPU)          |
| `logger.py`   | Logging setup                        |
| `registry.py` | Optional dynamic registration system |

---

### 11.3 Data (`src/data/`)

Data handling pipeline.

| File            | Purpose                                        |
| --------------- | ---------------------------------------------- |
| `dataset.py`    | Dataset definition                             |
| `datamodule.py` | DataLoader creation                            |
| `transforms.py` | Data augmentation                              |
| `sampler.py`    | Sampling strategies (e.g., imbalance handling) |
| `split.py`      | Dataset splitting                              |

---

### 11.4 Models (`src/models/`)

Model definitions.

| Component       | Purpose                                |
| --------------- | -------------------------------------- |
| `base.py`       | Base model interface                   |
| `classifier.py` | Classification wrapper                 |
| `backbones/`    | Feature extractors (ResNet, ViT, etc.) |
| `heads/`        | Task-specific heads                    |
| `losses/`       | Loss functions                         |

---

### 11.5 Training (`src/train/`)

Training logic.

| File                | Purpose                    |
| ------------------- | -------------------------- |
| `trainer.py`        | Training controller        |
| `loop.py`           | Training/validation loops  |
| `optimizer.py`      | Optimizer setup            |
| `scheduler.py`      | Learning rate scheduling   |
| `early_stopping.py` | Early stopping             |
| `ema.py`            | Exponential Moving Average |

---

### 11.6 Distributed (`src/distributed/`)

Multi-GPU training.

| File             | Purpose                         |
| ---------------- | ------------------------------- |
| `setup.py`       | Initialize distributed training |
| `utils.py`       | Rank and world size helpers     |
| `sync.py`        | Metric synchronization          |
| `ddp_trainer.py` | DDP-specific training logic     |

---

### 11.7 Evaluation (`src/eval/`)

Model evaluation.

| File                | Purpose                |
| ------------------- | ---------------------- |
| `metrics.py`        | Evaluation metrics     |
| `confusion.py`      | Confusion matrix       |
| `classification.py` | Classification reports |
| `calibration.py`    | Model calibration      |

---

### 11.8 Inference (`src/inference/`)

Prediction pipelines.

| File               | Purpose                 |
| ------------------ | ----------------------- |
| `predict.py`       | Single image prediction |
| `batch_predict.py` | Batch inference         |
| `postprocess.py`   | Output processing       |

---

### 11.9 Explainability (`src/explainability/`)

Model interpretation tools.

| File          | Purpose                |
| ------------- | ---------------------- |
| `gradcam.py`  | Grad-CAM visualization |
| `scorecam.py` | Score-CAM              |
| `saliency.py` | Saliency maps          |

---

### 11.10 Visualization (`src/visualization/`)

Analysis and plotting.

| File                  | Purpose             |
| --------------------- | ------------------- |
| `training_curves.py`  | Loss/accuracy plots |
| `confusion_matrix.py` | Visualization       |
| `data_preview.py`     | Dataset inspection  |
| `misclassified.py`    | Error analysis      |

---

### 11.11 Utilities (`src/utils/`)

General helper functions.

| File                | Purpose                       |
| ------------------- | ----------------------------- |
| `checkpoint.py`     | Save/load checkpoints         |
| `experiment.py`     | Experiment directory creation |
| `metrics_logger.py` | Save metrics                  |
| `metrics_utils.py`  | Metric helpers                |
| `image_ops.py`      | Image utilities               |
| `files.py`          | File handling                 |
| `io.py`             | Input/output operations       |
| `misc.py`           | Miscellaneous utilities       |

---

## 12. Standard Workflow

1. Prepare dataset in `data/`
2. Define configs in `configs/`
3. Run training:

   ```bash
   make ddp
   ```
4. Monitor logs in `experiments/`
5. Analyze results (metrics, Grad-CAM, confusion matrix)
6. Iterate via config changes

---

## 13. Best Practices

* Do not modify raw data
* Always log configurations
* Use fixed seeds for experiments
* Save all checkpoints and metrics
* Avoid hardcoding parameters
* Use version control for code changes

---

## 14. Common Pitfalls

* Logging from all DDP processes
* Not saving configs
* Mixing model and training logic
* Ignoring class imbalance
* Overwriting experiments

---

## 15. Extending the Template

To add new functionality:

* New dataset → `src/data/`
* New model → `src/models/`
* New loss → `src/models/losses/`
* New metric → `src/eval/`

---

## 16. Conclusion

This template provides a robust foundation for:

* academic research
* experimentation
* production prototyping

Users are encouraged to maintain modularity, reproducibility, and clarity while extending the framework.
