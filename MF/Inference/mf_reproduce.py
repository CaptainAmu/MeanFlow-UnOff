"""
MeanFlow Sampling Reproduction: Compare 1-Step vs Multi-Step ODE Integration
============================================================================

This script loads a trained MeanFlow checkpoint and compares two sampling strategies
from the same initial noise points:

1. **MeanFlow 1-step (t=1→0)**: Uses the average predicted velocity u(z, 1, 0) to
   transport noise to data in a single step. This exploits the flow-matching property
   that u(z, t, r) approximates the average velocity from t to r.

2. **Multi-step ODE integration**: Uses the instantaneous velocity u(z, t, t) with
   either Euler or Heun method over several steps. Euler: z_{k+1} = z_k - dt * u(z_k, t_k, t_k).
   Heun (2nd-order): predictor-corrector using u(z_k, t_k, t_k) and u(z_pred, t_{k+1}, t_{k+1}).

Both methods start from the same Gaussian noise z ~ N(0,I) conditioned on class labels.
Paired results are plotted side-by-side and saved to MF_reproduce/ for visual comparison.

Usage:
    python Inference/mf_reproduce.py --ckpt checkpoints/step_100000.pt [options]
    (from MF/; or use sbatch Inference/mf_reproduce.job)
"""

import argparse
import os
import sys
from pathlib import Path

# Resolve imports: models from project root, meanflow from Train/
_script_dir = Path(__file__).resolve().parent
_mf_root = _script_dir.parent
_project_root = _mf_root.parent
_train_dir = _mf_root / "Train"
for p in (_project_root, _train_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from torchvision.utils import make_grid, save_image
from einops import rearrange

from models.dit import MFDiT
from meanflow import MeanFlow


def sample_1step_meanflow(model, meanflow, z0, c, device):
    """
    Method 1: Single step from t=1 to t=0 using u(z, 1, 0).
    z_final = z0 - (1 - 0) * u(z0, 1, 0)
    """
    t = torch.full((z0.size(0),), 1.0, device=device)
    r = torch.full((z0.size(0),), 0.0, device=device)
    u = model(z0, t, r, c)
    z = z0 - 1.0 * u
    return meanflow.normer.unnorm(z)


def sample_euler(model, meanflow, z0, c, device, steps=10):
    """
    Method 2a: Euler integration using u(z, t, t) as instantaneous velocity.
    """
    z = z0.clone()
    t_vals = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = torch.full((z.size(0),), t_vals[i].item(), device=device)
        r_val = t_vals[i + 1].item()
        r = torch.full((z.size(0),), r_val, device=device)
        v = model(z, t, t, c)
        dt = t_vals[i].item() - r_val
        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")
        z = z - (t_ - r_) * v
    return meanflow.normer.unnorm(z)


def sample_heun(model, meanflow, z0, c, device, steps=10):
    """
    Method 2b: Heun (2nd-order) integration using u(z, t, t).
    Predictor: z_pred = z - dt * v(z, t, t)
    Corrector: z_new = z - dt * (v(z,t,t) + v(z_pred, r, r)) / 2
    """
    z = z0.clone()
    t_vals = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = torch.full((z.size(0),), t_vals[i].item(), device=device)
        r_val = t_vals[i + 1].item()
        r = torch.full((z.size(0),), r_val, device=device)
        dt = t_vals[i].item() - r_val
        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        v1 = model(z, t, t, c)
        z_pred = z - (t_ - r_) * v1
        v2 = model(z_pred, r, r, c)
        z = z - (t_ - r_) * (v1 + v2) / 2
    return meanflow.normer.unnorm(z)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce MeanFlow sampling: compare 1-step vs Euler/Heun multi-step."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default="MF_reproduce", help="Output directory")
    parser.add_argument("--n_per_class", type=int, default=2, help="Samples per class")
    parser.add_argument("--ode_steps", type=int, default=10, help="ODE steps for Euler/Heun")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "mnist":
        channels, num_classes = 1, 10
    else:
        channels, num_classes = 3, 10

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["state_dict"]
    model_config = ckpt["model_config"]
    meanflow_config = ckpt["meanflow_config"]

    model = MFDiT(**model_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    meanflow = MeanFlow(**meanflow_config)

    # Shared initial noise: n_per_class samples for each of num_classes
    c = torch.arange(num_classes, device=device).repeat(args.n_per_class)
    z0 = torch.randn(
        c.shape[0], channels, meanflow.image_size, meanflow.image_size,
        device=device
    )

    with torch.no_grad():
        # Method 1: 1-step MeanFlow
        z_1step = sample_1step_meanflow(model, meanflow, z0, c, device)

        # Method 2: Euler and/or Heun
        z_euler = sample_euler(model, meanflow, z0, c, device, args.ode_steps)
        z_heun = sample_heun(model, meanflow, z0, c, device, args.ode_steps)

    n_total = z_1step.size(0)

    # Build paired grid: 1-step vs Euler (always)
    paired = torch.stack([z_1step, z_euler], dim=1).view(n_total * 2, *z_1step.shape[1:])
    save_image(make_grid(paired, nrow=2),
               os.path.join(args.output_dir, "paired_1step_vs_euler.png"))
    print(f"Saved paired 1-step vs Euler to {args.output_dir}/paired_1step_vs_euler.png")

    # 1-step vs Heun
    paired_heun = torch.stack([z_1step, z_heun], dim=1).view(n_total * 2, *z_1step.shape[1:])
    save_image(make_grid(paired_heun, nrow=2),
               os.path.join(args.output_dir, "paired_1step_vs_heun.png"))
    print(f"Saved paired 1-step vs Heun to {args.output_dir}/paired_1step_vs_heun.png")

    # Euler vs Heun
    paired_eh = torch.stack([z_euler, z_heun], dim=1).view(n_total * 2, *z_1step.shape[1:])
    save_image(make_grid(paired_eh, nrow=2),
               os.path.join(args.output_dir, "paired_euler_vs_heun.png"))
    print(f"Saved paired Euler vs Heun to {args.output_dir}/paired_euler_vs_heun.png")

    # Separate grids per method
    save_image(make_grid(z_1step, nrow=num_classes),
               os.path.join(args.output_dir, "samples_1step.png"))
    save_image(make_grid(z_euler, nrow=num_classes),
               os.path.join(args.output_dir, f"samples_euler{args.ode_steps}step.png"))
    save_image(make_grid(z_heun, nrow=num_classes),
               os.path.join(args.output_dir, f"samples_heun{args.ode_steps}step.png"))
    print(f"Saved all grids to {args.output_dir}/")


if __name__ == "__main__":
    main()
