# """
# Configuration loading and merging with OmegaConf.
# Loads YAML configs and applies CLI overrides.
# """

# from omegaconf import OmegaConf, DictConfig
# from pathlib import Path
# from typing import Optional, List


# def load_config(
#     config_dir: Path = Path("configs"),
#     overrides: Optional[List[str]] = None,
# ) -> DictConfig:
#     """
#     Load and merge all configs from configs/ directory.
    
#     Order of merging (later overrides earlier):
#     1. default.yaml
#     2. data/pests.yaml
#     3. model/*.yaml (dinov2, arcface, sam, heads)
#     4. train/*.yaml (phase1, phase2, phase3, ddp)
#     5. CLI overrides
    
#     Args:
#         config_dir: Path to configs directory
#         overrides: List of CLI overrides (e.g., ["learning_rate=1e-4", "seed=42"])
    
#     Returns:
#         Merged DictConfig
#     """
#     config_dir = Path(config_dir)
    
#     if not config_dir.exists():
#         raise FileNotFoundError(f"Config directory not found: {config_dir}")

#     # Load base config
#     default_cfg = OmegaConf.load(config_dir / "default.yaml")
    
#     # Load data config
#     data_cfg = OmegaConf.load(config_dir / "data" / "pests.yaml")
    
#     # Load all model configs
#     model_configs = [
#         OmegaConf.load(config_dir / "model" / "dinov2.yaml"),
#         OmegaConf.load(config_dir / "model" / "arcface.yaml"),
#         OmegaConf.load(config_dir / "model" / "sam.yaml"),
#         OmegaConf.load(config_dir / "model" / "heads.yaml"),
#     ]
#     model_cfg = OmegaConf.merge(*model_configs)
    
#     # Load all train configs
#     train_configs = [
#         OmegaConf.load(config_dir / "train" / "phase1.yaml"),
#         OmegaConf.load(config_dir / "train" / "phase2.yaml"),
#         OmegaConf.load(config_dir / "train" / "phase3.yaml"),
#         OmegaConf.load(config_dir / "train" / "ddp.yaml"),
#     ]
#     train_cfg = OmegaConf.merge(*train_configs)
    
#     # Merge everything
#     cfg = OmegaConf.merge(default_cfg, data_cfg, model_cfg, train_cfg)
    
#     # Apply CLI overrides
#     if overrides:
#         cfg = OmegaConf.update(cfg, overrides)
    
#     # Validate required fields
#     required_fields = ["seed", "device", "data_root", "labels_csv", "output_dir"]
#     for field in required_fields:
#         if field not in cfg:
#             raise KeyError(f"Missing required config field: {field}")
    
#     return cfg


# def save_config(cfg: DictConfig, output_dir: Path) -> None:
#     """
#     Save config to experiment directory as YAML.
    
#     Args:
#         cfg: Configuration to save
#         output_dir: Directory to save to
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     config_path = output_dir / "config.yaml"
#     OmegaConf.save(cfg, config_path)


# def print_config(cfg: DictConfig) -> None:
#     """
#     Pretty-print configuration.
    
#     Args:
#         cfg: Configuration to print
#     """
#     print("\n" + "="*60)
#     print("Configuration")
#     print("="*60)
#     print(OmegaConf.to_yaml(cfg))
#     print("="*60 + "\n")

"""
Configuration loading and merging with OmegaConf.
Loads YAML configs and applies CLI overrides.
"""

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Optional, List


def load_config(
    config_dir: Path = Path("configs"),
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Load and merge all configs from configs/ directory.

    Order of merging (later overrides earlier):
    1. default.yaml
    2. data/pests.yaml
    3. model/*.yaml
    4. train/*.yaml
    5. CLI overrides

    Args:
        config_dir: Path to configs directory
        overrides: CLI overrides
                   Example:
                   ["training.learning_rate=1e-4", "seed=42"]

    Returns:
        Merged DictConfig
    """

    config_dir = Path(config_dir)

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    # ==========================================================
    # Load base config
    # ==========================================================
    default_cfg = OmegaConf.load(config_dir / "default.yaml")

    # ==========================================================
    # Load data config
    # ==========================================================
    data_cfg = OmegaConf.load(config_dir / "data" / "pests.yaml")

    # ==========================================================
    # Load model configs
    # ==========================================================
    model_configs = [
        OmegaConf.load(config_dir / "model" / "dinov2.yaml"),
        OmegaConf.load(config_dir / "model" / "arcface.yaml"),
        OmegaConf.load(config_dir / "model" / "sam.yaml"),
        OmegaConf.load(config_dir / "model" / "heads.yaml"),
    ]

    model_cfg = OmegaConf.merge(*model_configs)

    # ==========================================================
    # Load training configs
    # ==========================================================
    train_configs = [
        OmegaConf.load(config_dir / "train" / "phase1.yaml"),
        OmegaConf.load(config_dir / "train" / "phase2.yaml"),
        OmegaConf.load(config_dir / "train" / "phase3.yaml"),
        OmegaConf.load(config_dir / "train" / "ddp.yaml"),
    ]

    train_cfg = OmegaConf.merge(*train_configs)

    # ==========================================================
    # Merge all configs
    # ==========================================================
    cfg = OmegaConf.merge(
        default_cfg,
        data_cfg,
        model_cfg,
        train_cfg,
    )

    # ==========================================================
    # Apply CLI overrides
    # ==========================================================
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # ==========================================================
    # Validate required fields
    # ==========================================================
    required_fields = [
        "seed",
        "device",
        "paths.data_root",
        "paths.labels_csv",
        "paths.output_dir",
    ]

    for field in required_fields:
        if OmegaConf.select(cfg, field) is None:
            raise KeyError(f"Missing required config field: {field}")

    return cfg


def save_config(cfg: DictConfig, output_dir: Path) -> None:
    """
    Save config to experiment directory as YAML.

    Args:
        cfg: Configuration to save
        output_dir: Directory to save to
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"

    OmegaConf.save(cfg, config_path)


def print_config(cfg: DictConfig) -> None:
    """
    Pretty-print configuration.

    Args:
        cfg: Configuration to print
    """

    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")