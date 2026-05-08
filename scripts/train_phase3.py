"""
Backward-compatible wrapper for Phase 3 training.

Prefer:
  PYTHONPATH=src python scripts/train.py --phase 3 --phase1-checkpoint <ckpt>
"""

from scripts.train import main


if __name__ == "__main__":
    main()