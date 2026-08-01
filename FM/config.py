"""
Config for flow matching teacher training.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

DATASET_PRESETS = {
    "mnist": {
        "in_channels": 1,
        "channels": 1,
        "n_steps": 10000,
        "root": "../dataset/mnist",
        "transform": "mnist",
    },
    "cifar10": {
        "in_channels": 3,
        "channels": 3,
        "n_steps": 200000,
        "root": "../dataset/cifar",
        "transform": "cifar10",
    },
}

DEFAULTS = {
    "dataset": "cifar10",
    "input_size": 32,
    "patch_size": 2,
    "dim": 384,
    "depth": 12,
    "num_heads": 6,
    "num_classes": 10,
    "image_size": 32,
    "flow_schedule": "Reflow",
    "time_dist": ["uniform"],
    "normalizer": ["minmax", None, None],
    "batch_size": 32,
    "n_steps": 200000,
    "lr": 1.0e-4,
    "weight_decay": 0.0,
    "log_step": 500,
    "sample_step": 500,
    "ckpt_step": 5000,
    "num_workers": 4,
    "euler_steps": 20,
}


def _coerce_normalizer(val: Optional[List[Any]]) -> List[Any]:
    if val is None:
        return DEFAULTS["normalizer"]
    return [v if v is not None else None for v in val]


def load_config(path: str = "config_flow.yml", dataset_override: Optional[str] = None) -> "Config":
    cfg_dict = DEFAULTS.copy()
    yaml_cfg = {}

    if os.path.exists(path):
        with open(path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        for k, v in yaml_cfg.items():
            if k == "normalizer":
                cfg_dict[k] = _coerce_normalizer(v)
            else:
                cfg_dict[k] = v

    dataset = dataset_override or cfg_dict["dataset"]
    if dataset not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    preset = DATASET_PRESETS[dataset]
    cfg_dict["dataset"] = dataset
    cfg_dict["in_channels"] = preset["in_channels"]
    cfg_dict["channels"] = preset["channels"]
    cfg_dict["root"] = preset["root"]
    cfg_dict["transform"] = preset["transform"]
    if "n_steps" not in yaml_cfg:
        cfg_dict["n_steps"] = preset["n_steps"]

    return Config(**cfg_dict)


@dataclass
class Config:
    dataset: str
    input_size: int
    patch_size: int
    dim: int
    depth: int
    num_heads: int
    num_classes: int
    in_channels: int
    channels: int
    image_size: int
    flow_schedule: str
    time_dist: List[Any]
    normalizer: List[Any]
    root: str
    transform: str
    batch_size: int
    n_steps: int
    lr: float
    weight_decay: float
    log_step: int
    sample_step: int
    ckpt_step: int
    num_workers: int
    euler_steps: int

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
