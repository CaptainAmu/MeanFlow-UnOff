"""
Sample from a pretrained MeanFlow model checkpoint.
Usage: python Inference/sample.py --ckpt checkpoints/step_10000.pt [options]
       (from MF/; checkpoint path relative to MF/)
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
from models.dit import MFDiT
from meanflow import MeanFlow

def main():
    parser = argparse.ArgumentParser(description="Sample from a pretrained MeanFlow model checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--n_per_class", type=int, default=2, help="Samples per class")
    parser.add_argument("--use", type=str, default="u", choices=["u", "v"], help="Use u(z, t, r) or v(z, t, t) for sampling")
    parser.add_argument("--sample_steps", type=int, default=5, help="ODE integration steps (5 = 1-step-ish)")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="Dataset type (determines channels, num_classes)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset=="mnist":
        channels, num_classes = 1, 10
    else:
        channels, num_classes = 3, 10

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["state_dict"]
    model_config = ckpt["model_config"]
    meanflow_config = ckpt["meanflow_config"]
        

    model = MFDiT(**model_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    meanflow = MeanFlow(**meanflow_config)

    z = meanflow.sample_each_class(
        model,
        n_per_class = args.n_per_class,
        classes = None,
        sample_steps = args.sample_steps,
        device = device,
        use = args.use,
    )

    out_path = os.path.join(args.output_dir, "samples.png")
    save_image(make_grid(z, nrow=10), out_path)
    print(f"Saved {z.shape[0]} samples to {out_path}")

if __name__ == "__main__":
    main()