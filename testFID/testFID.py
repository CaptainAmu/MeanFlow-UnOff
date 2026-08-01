"""
Compute FID for FM and MF model checkpoints against CIFAR-10 train set.
Uses Inception v3 features (2048-dim) and Frechet distance.

Flow: 1) Ref embeddings  2) MF checkpoints (FID -> save)  3) FM checkpoints (FID -> append)

Required: pip install scipy  (before sbatch)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# #region agent log
def _dlog(hid, msg, **data):
    try:
        with open("/slurm-storage/shucli/PROJECT_FOLDER/.cursor/debug-59ea4a.log", "a") as f:
            f.write(json.dumps({"sessionId":"59ea4a","hypothesisId":hid,"message":msg,"data":data,"timestamp":int(time.time()*1000),"location":"testFID.py"}) + "\n")
    except Exception:
        pass
# #endregion
sys.path.insert(0, str(PROJECT_ROOT))

# Check scipy early
try:
    import scipy.linalg
except ImportError:
    print("ERROR: scipy is required for FID. Install with: pip install scipy", flush=True)
    sys.exit(1)

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_inception_features_model(device):
    """Inception v3 pretrained, returns 2048-dim features (before classifier)."""
    model = torchvision.models.inception_v3(weights="IMAGENET1K_V1")
    model.fc = nn.Identity()
    model.aux_logits = False
    model = model.to(device).eval()
    return model


def preprocess_for_inception(x):
    """Resize to 299x299 and ImageNet normalize for Inception v3."""
    x = torch.nn.functional.interpolate(
        x, size=(299, 299), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def compute_features(dataloader, model, device, max_samples=None, desc="Features"):
    """Extract Inception features, return (N, 2048) numpy array."""
    feats = []
    n = 0
    for batch in tqdm(dataloader, desc=desc, file=sys.stdout):
        x = batch[0].to(device)
        x = preprocess_for_inception(x)
        with torch.no_grad():
            f = model(x)
        feats.append(f.cpu().numpy())
        n += x.shape[0]
        if max_samples and n >= max_samples:
            break
    feats = np.concatenate(feats, axis=0)
    if max_samples:
        feats = feats[:max_samples]
    return feats.astype(np.float64)


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Frechet distance between two Gaussians."""
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(np.sum(diff**2) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def get_stats(feats):
    """Compute mu and sigma from features (N, D)."""
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


class Normalizer:
    def __init__(self, mode="minmax"):
        self.mode = mode

    def norm(self, x):
        return x * 2 - 1

    def unnorm(self, x):
        x = x.clip(-1, 1)
        return (x + 1) * 0.5


def generate_fm(model, normer, schedule, num_samples, batch_size, device, euler_steps):
    """Generate samples from FM (FlowMatchDiT) model. Same logic as sample_teacher sample_each_class."""
    from utils.flow_schedule import get_t_end, velocity_from_x0

    model.eval()
    t_end = get_t_end(schedule)
    samples = []
    n_done = 0
    num_classes = model.num_classes
    ch, sz = model.out_channels, model.input_size
    n_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for b in tqdm(range(n_batches), desc="FM gen", file=sys.stdout):
            bs = min(batch_size, num_samples - n_done)
            c = torch.randint(0, num_classes, (bs,), device=device)
            z = torch.randn(bs, ch, sz, sz, device=device)
            t_vals = torch.linspace(t_end, 0.0, euler_steps + 1, device=device)

            for i in range(euler_steps):
                t_i = t_vals[i]
                t_next = t_vals[i + 1]
                dt = t_i - t_next
                t_batch = torch.full((bs,), t_i, device=device, dtype=z.dtype)
                x0_hat = model(z, t_batch, c)
                v = velocity_from_x0(z, x0_hat, t_batch, schedule)
                z = z - dt * v

            x = normer.unnorm(z)
            samples.append(x.cpu())
            n_done += bs

    return torch.cat(samples, dim=0)[:num_samples]


def generate_mf(model, meanflow, num_samples, batch_size, device, sample_steps=5):
    """Generate samples from MF (MFDiT + MeanFlow) model. Uses average velocity u(z,t,r)."""
    from einops import rearrange

    model.eval()
    samples = []
    n_done = 0
    num_classes = meanflow.num_classes
    ch, sz = meanflow.channels, meanflow.image_size
    n_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="MF gen", file=sys.stdout):
            bs = min(batch_size, num_samples - n_done)
            c = torch.randint(0, num_classes, (bs,), device=device)
            z = torch.randn(bs, ch, sz, sz, device=device)
            t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

            for i in range(sample_steps):
                t = torch.full((bs,), t_vals[i], device=device)
                r = torch.full((bs,), t_vals[i + 1], device=device)
                v = model(z, t, r, c)  # average velocity u(z,t,r)
                t_ = rearrange(t, "b -> b 1 1 1")
                r_ = rearrange(r, "b -> b 1 1 1")
                z = z - (t_ - r_) * v

            x = meanflow.normer.unnorm(z)
            samples.append(x.cpu())
            n_done += bs

    return torch.cat(samples, dim=0)[:num_samples]


def load_fm_checkpoint(path, device):
    """Load FM (FlowMatchDiT) checkpoint."""
    from models.dit import FlowMatchDiT

    ckpt = torch.load(path, map_location="cpu")
    model = FlowMatchDiT(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    schedule = ckpt.get("flow_schedule", "Reflow")
    return model, schedule


def load_mf_checkpoint(path, device):
    """Load MF (MFDiT + MeanFlow) checkpoint."""
    from models.dit import MFDiT

    mf_train = PROJECT_ROOT / "MF" / "Train"
    if str(mf_train) not in sys.path:
        sys.path.insert(0, str(mf_train))
    from meanflow import MeanFlow

    ckpt = torch.load(path, map_location="cpu")
    model = MFDiT(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    meanflow = MeanFlow(**ckpt["meanflow_config"])
    return model, meanflow


def main():
    # #region agent log
    _dlog("H1", "main() entered", cwd=os.getcwd(), argv=sys.argv)
    # #endregion
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="testFID.yml")
    parser.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path")
    parser.add_argument("--model_type", type=str, default=None, choices=["FM", "MF"])
    args = parser.parse_args()

    import yaml
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    # #region agent log
    _dlog("H4", "config_path resolved", config_path=config_path, config_exists=os.path.exists(config_path), __file__=__file__)
    # #endregion
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    dataset_root = cfg.get("dataset_root", "../dataset/cifar")
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(PROJECT_ROOT, dataset_root)
    num_samples = int(cfg.get("num_samples", 50000))
    batch_size = int(cfg.get("batch_size", 64))
    euler_steps = int(cfg.get("euler_steps", 20))
    mf_sample_steps = int(cfg.get("mf_sample_steps", 5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Build checkpoints list
    if args.ckpt:
        checkpoints = [{"path": args.ckpt, "model_type": args.model_type or "FM", "name": os.path.basename(args.ckpt)}]
    else:
        checkpoints = cfg.get("checkpoints", [])

    if not checkpoints:
        for d, mtype in [
            (PROJECT_ROOT / "FM_Trained_Models", "FM"),
            (PROJECT_ROOT / "MF_Trained_Models", "MF"),
        ]:
            if d.exists():
                for f in sorted(d.rglob("*.pt")):
                    checkpoints.append({"path": str(f), "model_type": mtype, "name": f.name})

    if not checkpoints:
        # #region agent log
        _dlog("H2", "No checkpoints found, returning early", fm_dir=str(PROJECT_ROOT / "FM_Trained_Models"), mf_dir=str(PROJECT_ROOT / "MF_Trained_Models"))
        # #endregion
        print("No checkpoints found. Set checkpoints in testFID.yml or use --ckpt.")
        return

    # Split into MF and FM (process MF first, then FM)
    mf_items = [c for c in checkpoints if c["model_type"] == "MF"]
    fm_items = [c for c in checkpoints if c["model_type"] == "FM"]
    # #region agent log
    _dlog("H2", "checkpoints scanned", n_total=len(checkpoints), n_mf=len(mf_items), n_fm=len(fm_items), cwd=os.getcwd(), dataset_root=dataset_root)
    # #endregion
    print(f"Checkpoints: {len(mf_items)} MF, {len(fm_items)} FM (will process MF first, then FM)", flush=True)

    # ----- 1. Reference (CIFAR-10 train) embeddings -----
    transform = T.Compose([T.ToTensor()])
    ref_dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)
    ref_loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

    inception = get_inception_features_model(device)
    print("Step 1: Computing reference (CIFAR-10 train) features...", flush=True)
    ref_feats = compute_features(ref_loader, inception, device, max_samples=num_samples, desc="Ref")
    mu_ref, sigma_ref = get_stats(ref_feats)
    print(f"Reference: {len(ref_feats)} samples", flush=True)

    results = []  # [(name, model_type, fid), ...]

    # ----- 2. MF checkpoints (faster) - compute FID first -----
    for item in mf_items:
        path = item["path"]
        name = item.get("name", os.path.basename(path))
        if not os.path.exists(path):
            print(f"Skipping {name}: file not found", flush=True)
            continue

        print(f"\n--- {name} (MF) ---", flush=True)
        try:
            model, meanflow = load_mf_checkpoint(path, device)
            gen = generate_mf(model, meanflow, num_samples, batch_size, device, mf_sample_steps)
            gen_loader = DataLoader(
                torch.utils.data.TensorDataset(gen),
                batch_size=batch_size,
                shuffle=False,
            )
            gen_feats = compute_features(gen_loader, inception, device, max_samples=None, desc=f"Gen {name}")
            mu_gen, sigma_gen = get_stats(gen_feats)
            fid = frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
            print(f"FID: {fid:.4f}", flush=True)
            results.append((name, "MF", fid))
        except Exception as e:
            print(f"Error: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # ----- 3. FM checkpoints (slower) - compute FID second -----
    for item in fm_items:
        path = item["path"]
        name = item.get("name", os.path.basename(path))
        if not os.path.exists(path):
            print(f"Skipping {name}: file not found", flush=True)
            continue

        print(f"\n--- {name} (FM) ---", flush=True)
        try:
            model, schedule = load_fm_checkpoint(path, device)
            normer = Normalizer()
            gen = generate_fm(model, normer, schedule, num_samples, batch_size, device, euler_steps)
            gen_loader = DataLoader(
                torch.utils.data.TensorDataset(gen),
                batch_size=batch_size,
                shuffle=False,
            )
            gen_feats = compute_features(gen_loader, inception, device, max_samples=None, desc=f"Gen {name}")
            mu_gen, sigma_gen = get_stats(gen_feats)
            fid = frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
            print(f"FID: {fid:.4f}", flush=True)
            results.append((name, "FM", fid))
        except Exception as e:
            print(f"Error: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # #region agent log
    _dlog("H1", "main() completed", results_written=len(results))
    # #endregion
    print("\n" + "=" * 50, flush=True)
    print("FID Summary (output in fid_out.txt):", flush=True)
    print("name\tmodel_type\tfid", flush=True)
    for name, mtype, fid in results:
        print(f"{name}\t{mtype}\t{fid:.4f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    # #region agent log
    _dlog("H1", "script entry, about to call main()", __name__=__name__)
    # #endregion
    main()
