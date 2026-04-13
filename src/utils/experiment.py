import os
from datetime import datetime

def create_experiment_dir(base_dir="experiments", exp_name=None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if exp_name is None:
        exp_name = f"exp_{timestamp}"
    else:
        exp_name = f"{exp_name}_{timestamp}"

    exp_path = os.path.join(base_dir, exp_name)

    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_path, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(exp_path, "gradcam"), exist_ok=True)

    return exp_path

