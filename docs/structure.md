# Project Structure and Module Description

## 1. System Overview

The repository follows a modular pipeline for computer vision experimentation:

```
configs → data → src → experiments → outputs
```

- `configs`: Defines experiment parameters  
- `data`: Stores datasets  
- `src`: Contains implementation  
- `experiments`: Stores individual experiment results  
- `outputs`: Stores aggregated results  

---

## 2. Root-Level Directories

### 2.1 `configs/` — Experiment Configuration

**Purpose:** Centralized configuration for all experiments.

#### Subdirectories:
- `data/`: Dataset-specific configurations  
- `model/`: Model architecture configurations  
- `train/`: Training hyperparameters  
- `default.yaml`: Global defaults  

#### Usage:
- Modify model architectures (e.g., ResNet to ViT)  
- Tune hyperparameters  
- Conduct ablation studies  

**Guideline:**  
All experiment parameters must be defined here. Avoid hardcoding values in code.

---

### 2.2 `data/` — Dataset Storage

**Purpose:** Maintain datasets across different processing stages.

| Folder       | Description                        |
|--------------|------------------------------------|
| `raw/`       | Original dataset (immutable)       |
| `processed/` | Cleaned and preprocessed data     |
| `splits/`    | Train/validation/test partitions  |
| `metadata/`  | Class mappings and statistics     |

**Guideline:**  
Do not modify data in `raw/`.

---

### 2.3 `experiments/` — Experiment Tracking

**Purpose:** Store outputs of individual experiments.

Example:
```
experiments/resnet50_20260408/
```

Typical contents:
- Configuration used  
- Logs  
- Model checkpoints  
- Metrics  
- Predictions  
- Grad-CAM visualizations  

**Role:**  
Serves as the primary source for reproducibility and comparison.

---

### 2.4 `outputs/` — Aggregated Results

**Purpose:** Store final or combined outputs across experiments.

Includes:
- Comparative plots  
- Reports  
- Aggregated metrics  

---

### 2.5 `models/` — Model Artifacts

**Purpose:** Store model files, not source code.

| Folder        | Description                        |
|---------------|------------------------------------|
| `pretrained/` | External pretrained weights         |
| `exported/`   | Final exported models (ONNX, etc.) |

---

### 2.6 `scripts/` — Execution Utilities

**Purpose:** Simplify execution of common tasks.

| File              | Description              |
|-------------------|--------------------------|
| `run_ddp.sh`      | Multi-GPU training       |
| `run_train.sh`    | Single GPU training      |
| `prepare_data.py` | Data preprocessing       |

---

### 2.7 `docs/` — Documentation

Contains:
- SOP documents  
- Quick start guide  
- Design notes  

---

### 2.8 `tests/` — Testing Suite

**Purpose:** Validate correctness of components.

Typical coverage:
- Dataset loading  
- Model forward pass  
- Metrics  

---

## 3. Source Code (`src/`)

All implementation resides here.

---

### 3.1 `cli/` — Entry Points

**Purpose:** Interface for executing workflows.

| File          | Description              |
|---------------|--------------------------|
| `train.py`    | Training pipeline        |
| `evaluate.py` | Evaluation pipeline      |
| `infer.py`    | Inference pipeline       |
| `explain.py`  | Interpretability tools   |

---

### 3.2 `core/` — Infrastructure Layer

**Purpose:** Core utilities used across modules.

| File          | Description                     |
|---------------|---------------------------------|
| `config.py`   | Configuration management        |
| `seed.py`     | Reproducibility control         |
| `device.py`   | Device (CPU/GPU) handling       |
| `logger.py`   | Logging utilities               |
| `registry.py` | Dynamic module registration     |

---

### 3.3 `data/` — Data Pipeline

**Purpose:** Handle all data-related operations.

| File            | Description                |
|-----------------|----------------------------|
| `dataset.py`    | Dataset definition         |
| `datamodule.py` | DataLoader construction    |
| `transforms.py` | Data augmentation          |
| `sampler.py`    | Sampling strategies        |
| `split.py`      | Dataset splitting          |

---

### 3.4 `models/` — Model Definitions

**Purpose:** Define model architectures.

| Component        | Description                |
|------------------|----------------------------|
| `base.py`        | Base model abstraction     |
| `classifier.py`  | Classification wrapper     |
| `backbones/`     | Feature extractors         |
| `heads/`         | Task-specific heads        |
| `losses/`        | Loss functions             |

---

### 3.5 `train/` — Training Module

**Purpose:** Implement training logic.

| File                | Description                  |
|---------------------|------------------------------|
| `trainer.py`        | Training controller          |
| `loop.py`           | Training/validation loops    |
| `optimizer.py`      | Optimizer setup              |
| `scheduler.py`      | Learning rate scheduling     |
| `early_stopping.py` | Early stopping mechanism     |
| `ema.py`            | Model weight averaging       |

---

### 3.6 `distributed/` — Distributed Training

**Purpose:** Enable multi-GPU training.

| File             | Description                  |
|------------------|------------------------------|
| `setup.py`       | Initialize distributed mode  |
| `utils.py`       | Rank/world size utilities    |
| `sync.py`        | Metric synchronization       |
| `ddp_trainer.py` | Distributed training logic   |

---

### 3.7 `eval/` — Evaluation

**Purpose:** Compute performance metrics.

| File                | Description                  |
|---------------------|------------------------------|
| `metrics.py`        | Evaluation metrics           |
| `confusion.py`      | Confusion matrix computation |
| `classification.py` | Classification reports       |
| `calibration.py`    | Confidence calibration       |

---

### 3.8 `inference/` — Prediction

**Purpose:** Perform model inference.

| File               | Description             |
|--------------------|-------------------------|
| `predict.py`       | Single sample inference |
| `batch_predict.py` | Batch inference         |
| `postprocess.py`   | Output processing       |

---

### 3.9 `explainability/` — Model Interpretation

**Purpose:** Provide interpretability tools.

| File          | Description   |
|---------------|---------------|
| `gradcam.py`  | Grad-CAM      |
| `scorecam.py` | Score-CAM     |
| `saliency.py` | Saliency maps |

---

### 3.10 `visualization/` — Analysis Tools

**Purpose:** Visualize results and data.

| File                  | Description            |
|-----------------------|------------------------|
| `training_curves.py`  | Training plots         |
| `confusion_matrix.py` | Visualization          |
| `data_preview.py`     | Dataset inspection     |
| `misclassified.py`    | Error analysis         |

---

### 3.11 `utils/` — Utility Functions

**Purpose:** Provide reusable helper functions.

| File                | Description                   |
|---------------------|-------------------------------|
| `checkpoint.py`     | Model checkpointing           |
| `experiment.py`     | Experiment directory creation |
| `metrics_logger.py` | Metric logging                |
| `metrics_utils.py`  | Metric utilities              |
| `image_ops.py`      | Image operations              |
| `files.py`          | File handling                 |
| `io.py`             | Input/output operations       |
| `misc.py`           | Miscellaneous helpers         |

---

## 4. Execution Flow

When executing:

```bash
make ddp
```

The following sequence occurs:

```
cli/train.py
   ↓
Load configurations
   ↓
Initialize data pipeline
   ↓
Initialize model
   ↓
Training loop execution
   ↓
Distributed synchronization
   ↓
Evaluation
   ↓
Logging and checkpointing
   ↓
Save outputs to experiments/
```

---

## 5. Summary

This structure ensures:

- Modularity  
- Reproducibility  
- Scalability  
- Clear separation of responsibilities  

It is suitable for:

- Academic research  
- Experimental workflows  
- Production prototyping  