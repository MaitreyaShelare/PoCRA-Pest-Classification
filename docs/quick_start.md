# 🚀 Quick Start Guide  
## CV Research Template (PyTorch + DDP)

---

## 1. Setup Environment

### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Project Structure (What matters right now)

```
configs/        → experiment settings  
data/           → dataset  
src/            → code  
scripts/        → run commands  
experiments/    → results (auto-created)  
```

👉 You mainly work with:
- `configs/`
- `data/`
- `src/data/` (for custom datasets)

---

## 3. Prepare Dataset

### Step 1: Place raw data
```
data/raw/
```

### Step 2: Process data (optional)
```bash
python scripts/prepare_data.py
```

### Step 3: Create splits
```
data/splits/train.txt  
data/splits/val.txt  
data/splits/test.txt  
```

Each line format:
```
path/to/image.jpg label
```

---

## 4. Configure Experiment

### Dataset config
```yaml
# configs/data/pests.yaml
dataset_path: data/processed
num_classes: 10
img_size: 224
```

### Model config
```yaml
# configs/model/resnet50.yaml
name: resnet50
pretrained: true
num_classes: 10
```

### Training config
```yaml
# configs/train/ddp.yaml
batch_size: 32
epochs: 20
lr: 1e-3
```

---

## 5. Run Training

### Single GPU
```bash
bash scripts/run_train.sh
```

### Multi-GPU (DDP)
```bash
bash scripts/run_ddp.sh
```

Or:
```bash
make ddp
```

---

## 6. Outputs (Where to look)

After training:

```
experiments/
└── exp_<name>_<timestamp>/
    ├── config.yaml
    ├── logs/train.log
    ├── checkpoints/
    ├── metrics.json
    ├── predictions/
    └── gradcam/
```

---

## 7. Evaluate Model

```bash
PYTHONPATH=src python src/cli/evaluate.py
```

---

## 8. Run Inference

```bash
PYTHONPATH=src python src/cli/infer.py
```

---

## 9. Generate Grad-CAM

```bash
PYTHONPATH=src python src/cli/explain.py
```

Outputs saved in:
```
experiments/.../gradcam/
```

---

## 10. Debug Mode (Recommended)

```yaml
# configs/train/debug.yaml
epochs: 1
batch_size: 2
```

---

## 11. Common Issues

### Import errors
```bash
PYTHONPATH=src python src/cli/train.py
```

### DDP issues
- Ensure GPUs are available
- Use `torchrun`, not `python`

### No logs/checkpoints
- Only rank 0 saves in DDP

---

## 12. Typical Workflow

```
1. Prepare dataset
2. Edit config
3. Run training
4. Check metrics + logs
5. Analyze errors (Grad-CAM, confusion matrix)
6. Modify config
7. Repeat
```

---

## 13. Minimal Example (TL;DR)

```bash
pip install -r requirements.txt
make ddp
ls experiments/
```

---

## 14. Pro Tips

- Always save configs (automatic)
- Use meaningful experiment names
- Track results in `notes.md`
- Start with debug runs
- Use Grad-CAM to validate model behavior

---

## 15. Next Steps

- Add new model → `src/models/`
- Add new dataset → `src/data/`
- Add new loss → `src/models/losses/`
- Add new metric → `src/eval/`

---

## ✅ You’re Ready

You can now:
- Run experiments
- Track results
- Scale to multi-GPU
- Extend for research