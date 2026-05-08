"""
Backward-compatible wrapper.

Prefer:
  PYTHONPATH=src python scripts/train.py --phase 1
"""

from scripts.train import main


if __name__ == "__main__":
    main(["--phase", "1"])