"""
Experiment management: directory creation, metadata tracking, git versioning.
Essential for research reproducibility and ablation studies.
"""

from pathlib import Path
from datetime import datetime
import json
import subprocess
from typing import Optional, Dict, Any
from omegaconf import DictConfig, OmegaConf


def create_experiment_dir(output_dir: Path, experiment_name: str) -> Path:
    """
    Create experiment directory with timestamp.
    
    Args:
        output_dir: Base output directory
        experiment_name: Name of experiment (e.g., "phase1_arcface")
    
    Returns:
        Path to created experiment directory
        Format: outputs/experiment_name_YYYYMMDD_HHMMSS/
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def save_experiment_metadata(
    exp_dir: Path,
    cfg: DictConfig,
    git_commit: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save experiment metadata: config, git commit, timestamp, notes.
    
    Args:
        exp_dir: Experiment directory
        cfg: Configuration object
        git_commit: Git commit hash (auto-detected if None)
        additional_info: Extra metadata to save
    """
    exp_dir = Path(exp_dir)
    
    # Config already saved by config.py, but save again here
    OmegaConf.save(cfg, exp_dir / "config.yaml")
    
    # Get git commit
    if git_commit is None:
        git_commit = get_git_commit()
    
    # Get git status (uncommitted changes)
    git_status = get_git_status()
    
    # Build metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "git_status": git_status,
        "git_branch": get_git_branch(),
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    # Save metadata
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def get_git_commit() -> str:
    """
    Get current git commit hash.
    
    Returns:
        Commit hash, or "unknown" if not in git repo
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit
    except Exception:
        return "unknown"


def get_git_branch() -> str:
    """
    Get current git branch name.
    
    Returns:
        Branch name, or "unknown" if not in git repo
    """
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return branch
    except Exception:
        return "unknown"


def get_git_status() -> bool:
    """
    Check if there are uncommitted changes.
    
    Returns:
        True if there are uncommitted changes, False if clean
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return False


def save_experiment_notes(exp_dir: Path, notes: str) -> None:
    """
    Save experiment notes (human-readable).
    
    Args:
        exp_dir: Experiment directory
        notes: Free-form text notes about the experiment
    """
    exp_dir = Path(exp_dir)
    with open(exp_dir / "notes.txt", "w") as f:
        f.write(notes)


def load_experiment_metadata(exp_dir: Path) -> Dict[str, Any]:
    """
    Load metadata from completed experiment.
    
    Args:
        exp_dir: Experiment directory
    
    Returns:
        Metadata dict
    """
    exp_dir = Path(exp_dir)
    with open(exp_dir / "metadata.json") as f:
        return json.load(f)