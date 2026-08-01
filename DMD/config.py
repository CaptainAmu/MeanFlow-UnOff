"""
Config for DMD2 student training.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

DEFAULTS = {
    "teacher_path": "",
    "flow_schedule": "Reflow",
    "input_size": 32,
    "patch_size": 2,
    "dim": 384,
    "depth": 12,
    "num_heads": 6,
    "num_classes": 10,
    "in_channels": 3,
    "sigma_data": 0.5,
    "lambda_gan": 0.1,
    "n_disc_updates": 2,
    "n_score_updates": 5,
    "lr_score": 2.0e-4,
    "lambda_denoise": 0.5,
    "batch_size": 32,
    "n_steps": 50000,
    "lr_gen": 2.0e-4,
    "lr_disc": 2.0e-4,
    "weight_decay": 0.01,
    "log_step": 100,
    "sample_step": 500,
    "ckpt_step": 5000,
    "num_workers": 4,
    "root": "../dataset/cifar",
    "use_wandb": True,
    "wandb_project": "MF_DMD_CIFAR10",
    "wandb_entity": "",
    "wandb_run_name": "",
    "wandb_log_step": 10,
}


def load_config(path: str = "train_dmd_student.yml", teacher_override: Optional[str] = None) -> "Config":
    cfg_dict = DEFAULTS.copy()
    if os.path.exists(path):
        with open(path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        for k, v in yaml_cfg.items():
            cfg_dict[k] = v
    if teacher_override:
        cfg_dict["teacher_path"] = teacher_override
    return Config(**cfg_dict)


@dataclass
class Config:
    teacher_path: str
    flow_schedule: str
    input_size: int
    patch_size: int
    dim: int
    depth: int
    num_heads: int
    num_classes: int
    in_channels: int
    sigma_data: float
    lambda_gan: float
    n_disc_updates: int
    n_score_updates: int
    lr_score: float
    lambda_denoise: float
    batch_size: int
    n_steps: int
    lr_gen: float
    lr_disc: float
    weight_decay: float
    log_step: int
    sample_step: int
    ckpt_step: int
    num_workers: int
    root: str
    use_wandb: bool
    wandb_project: str
    wandb_entity: str
    wandb_run_name: str
    wandb_log_step: int

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "dim": self.dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "num_classes": self.num_classes,
        }
