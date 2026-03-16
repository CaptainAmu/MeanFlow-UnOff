"""
Config class for MeanFlow training. Loads from config.yml and applies dataset presets.
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

DATASET_PRESETS = {
    "mnist": {
        "in_channels": 1,
        "channels": 1,
        "n_steps": 10000,
        "root": "../../dataset/mnist",
        "transform": "mnist",
    },
    "cifar10": {
        "in_channels": 3,
        "channels": 3,
        "n_steps": 200000,
        "root": "../../dataset/cifar",
        "transform": "cifar10",
    },
}

DEFAULTS = {
    "dataset": "mnist",
    "input_size": 32,
    "patch_size": 2,
    "dim": 384,
    "depth": 12,
    "num_heads": 6,
    "num_classes": 10,
    "image_size": 32,
    "flow_ratio": 0.50,
    "time_dist": ["lognorm", -0.4, 1.0],
    "cfg_ratio": 0.10,
    "cfg_scale": 2.0,
    "cfg_uncond": "u",
    "normalizer": ["minmax", None, None],
    "batch_size": 48,
    "n_steps": 10000,
    "lr": 1.0e-4,
    "weight_decay": 0.0,
    "log_step": 500,
    "sample_step": 500,
    "ckpt_step": 5000,
    "num_workers": 4,
    "num_steps": 5,
    "sample_use": "u",
}


def _coerce_normalizer(val: List[Any]) -> List[Any]:
    """Convert YAML null to Python None in normalizer list."""
    if val is None:
        return DEFAULTS["normalizer"]
    return [v if v is not None else None for v in val]


def load_config(path: str = "config.yml", dataset_override: Optional[str] = None) -> "Config":
    """
    Load config from YAML, merge with defaults, apply dataset preset.
    """
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
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_PRESETS)}")

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
    """Config container with model_config and meanflow_config for construction."""

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
    flow_ratio: float
    time_dist: List[Any]
    cfg_ratio: float
    cfg_scale: float
    cfg_uncond: str
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
    num_steps: int
    sample_use: str

    @property
    def model_config(self) -> Dict[str, Any]:
        """Dict for MFDiT(**model_config)."""
        return {
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "dim": self.dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "num_classes": self.num_classes,
        }

    @property
    def meanflow_config(self) -> Dict[str, Any]:
        """Dict for MeanFlow(**meanflow_config)."""
        return {
            "channels": self.channels,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "normalizer": self.normalizer,
            "flow_ratio": self.flow_ratio,
            "time_dist": self.time_dist,
            "cfg_ratio": self.cfg_ratio,
            "cfg_scale": self.cfg_scale,
            "cfg_uncond": self.cfg_uncond,
            "jvp_api": "autograd",
        }
