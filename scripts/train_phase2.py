"""
Backward-compatible wrapper for Phase 2 training.

Prefer:
  PYTHONPATH=src python scripts/train.py --phase 2 --phase1-checkpoint <ckpt>
"""

from scripts.train import main


if __name__ == "__main__":
    main()