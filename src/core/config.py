import yaml
import os

def save_config(cfg: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

