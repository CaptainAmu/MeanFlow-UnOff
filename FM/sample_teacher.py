"""
Sample from a trained FM teacher using Euler ODE integration.
Loads checkpoint from FM/checkpoints, runs num_steps Euler steps, saves visualization.
Config: sample_teacher.yml (ckpt_path, num_steps, etc.)
"""
import argparse
import os

import yaml
import torch
from torchvision.utils import make_grid, save_image

from models.dit import FlowMatchDiT
from train_teacher import Normalizer, sample_each_class


def load_sample_config(path: str = "sample_teacher.yml") -> dict:
    """Load sample config from yaml. Returns dict with defaults."""
    defaults = {
        "ckpt_path": "checkpoints/step_100000.pt",
        "num_steps": 20,
        "n_per_class": 1,
        "seed": 42,
    }
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        defaults.update(cfg)
    return defaults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sample_teacher.yml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--n_per_class", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_sample_config(args.config)
    if args.ckpt is not None:
        cfg["ckpt_path"] = args.ckpt
    if args.num_steps is not None:
        cfg["num_steps"] = args.num_steps
    if args.n_per_class is not None:
        cfg["n_per_class"] = args.n_per_class
    if args.output is not None:
        cfg["output"] = args.output
    if args.seed is not None:
        cfg["seed"] = args.seed

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = cfg["ckpt_path"]
    if not os.path.isabs(ckpt_path):
        fm_root = os.path.dirname(os.path.abspath(args.config))
        ckpt_path = os.path.join(fm_root, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_config = ckpt['model_config']
    schedule = ckpt['flow_schedule']

    model = FlowMatchDiT(**model_config).to(device)
    model.load_state_dict(ckpt['state_dict'], strict = True)
    model.eval()

    normer = Normalizer(mode="minmax")
    samples = sample_each_class(
        model, normer, schedule,
        n_per_class=cfg["n_per_class"],
        device=device,
        euler_steps=cfg["num_steps"],
    )
    grid = make_grid(samples, nrow=model.num_classes)
    _output_dir = cfg.get("output") or "sample_images"
    output_path = os.path.join(_output_dir, f"fm_euler_{cfg['num_steps']}steps.png")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image(grid, output_path)   
    print(f"Saved {samples.shape[0]} samples to {output_path}")


if __name__ == "__main__":
    main()