"""
DMD2 student distillation: one-step generator from flow matching teacher.
Student = FlowMatchDiT, same architecture as teacher, init from teacher.
"""
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import argparse
import glob
import time
import os
import numpy as np
from typing import Optional

try:
    import wandb
except ImportError:
    wandb = None

from models.dit import FlowMatchDiT


def _remove_old_checkpoints(ckpt_dir, keep_path, pattern="student_step_*.pt"):
    """Remove all checkpoints matching pattern except keep_path."""
    for f in glob.glob(os.path.join(ckpt_dir, pattern)):
        if os.path.abspath(f) != os.path.abspath(keep_path):
            try:
                os.remove(f)
            except OSError:
                pass
from config import load_config
from utils.flow_schedule import get_alpha_sigma_t, interpolate, get_t_end
from dmd2 import (
    BottleneckDiscriminator,
    distribution_matching_loss,
    denoising_loss_fake,
    softplus_d_loss,
)


class Normalizer:
    def __init__(self, mode="minmax", mean=None, std=None):
        self.mode = mode
        if mode == "mean_std":
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    def norm(self, x):
        if self.mode == "minmax":
            return x * 2 - 1
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == "minmax":
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        return x * self.std.to(x.device) + self.mean.to(x.device)


def get_dataset(cfg):
    transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
    dataset = torchvision.datasets.CIFAR10(
        root=cfg.root, train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )


def one_step_generate(student, z, c, t_end, device):
    """G(z, c) = student(z, t_end, c)."""
    t = torch.full((z.shape[0],), t_end, device=device, dtype=z.dtype)
    return student(z, t, c)


def setup_wandb(cfg, output_dir: str) -> Optional[object]:
    """Initialize WandB run from config/env if enabled."""
    use_wandb = bool(getattr(cfg, "use_wandb", True))
    if not use_wandb:
        return None
    if wandb is None:
        print("WandB disabled: package not installed.")
        return None

    project = os.environ.get("WANDB_PROJECT", getattr(cfg, "wandb_project", "MF_DMD_CIFAR10"))
    entity = os.environ.get("WANDB_ENTITY", getattr(cfg, "wandb_entity", ""))
    run_name = os.environ.get("WANDB_RUN_NAME", getattr(cfg, "wandb_run_name", ""))
    if not run_name:
        run_name = os.path.basename(os.path.abspath(output_dir))

    init_kwargs = {
        "project": project,
        "name": run_name,
        "dir": output_dir,
    }
    if entity:
        init_kwargs["entity"] = entity

    run = wandb.init(**init_kwargs, config=vars(cfg))
    # Define metrics so they appear in Charts with step on x-axis
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_dmd_student.yml")
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--flow_schedule", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(
        path=args.config,
        teacher_override=args.teacher or os.environ.get("TEACHER"),
    )
    teacher_path = cfg.teacher_path
    if not teacher_path:
        raise ValueError(
            "Provide teacher path via --teacher, TEACHER env, or teacher_path in train_dmd_student.yml"
        )
    output_dir = args.output_dir or os.environ.get("DMD_OUTPUT_DIR", ".")
    images_dir = os.path.join(output_dir, "images")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    wandb_run = setup_wandb(cfg, output_dir)
    wandb_log_step = int(getattr(cfg, "wandb_log_step", cfg.log_step))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load teacher
    ckpt = torch.load(teacher_path, map_location="cpu")
    # flow_schedule: 1) from checkpoint, 2) from --flow_schedule / FLOW_SCHEDULE env, 3) from yml
    schedule = (
        ckpt.get("flow_schedule")
        or args.flow_schedule
        or os.environ.get("FLOW_SCHEDULE")
        or cfg.flow_schedule
    )
    print(f"Using flow_schedule: {schedule}")
    t_end = get_t_end(schedule)
    normer = Normalizer(mode="minmax")
    teacher = FlowMatchDiT(**ckpt["model_config"])
    teacher.load_state_dict(ckpt["state_dict"], strict=True)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student one step generator (G_theta): same arch, copy teacher weights
    student = FlowMatchDiT(**cfg.model_config)
    student.load_state_dict(ckpt["state_dict"], strict=True)
    student = student.to(device).train()

    # Fake score model (μ_fake): separate model to estimate score of fake distribution (DMD paper)
    fake_score_model = FlowMatchDiT(**ckpt["model_config"])
    fake_score_model.load_state_dict(ckpt["state_dict"], strict=True)
    fake_score_model = fake_score_model.to(device).train()

    discriminator = BottleneckDiscriminator(
        student, dim=cfg.dim, patch_size=cfg.patch_size
    ).to(device)

    opt_gen = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.lr_gen,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )
    opt_disc = torch.optim.AdamW(
        discriminator.cls_pred_branch.parameters(),
        lr=cfg.lr_disc,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )
    opt_score = torch.optim.AdamW(
        fake_score_model.parameters(),
        lr=cfg.lr_score,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )

    dataloader = get_dataset(cfg)
    real_iter = iter(dataloader)

    global_step = 0
    log_path = os.path.join(output_dir, "log.txt")

    with tqdm(range(cfg.n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("DMD2 Student")
        for step in pbar:
            bs = cfg.batch_size

            # ----- A. Generate a batch of images -----
            z = torch.randn(bs, 3, 32, 32, device=device)
            c = torch.randint(0, cfg.num_classes, (bs,), device=device, dtype=torch.long)
            x0 = one_step_generate(student, z, c, t_end, device)

            # ----- B. Update Generator G_theta, while fixing student score function μ_fake -----
            eps = torch.randn_like(x0, device=device)
            # Restrict t to [0.02, 0.98] to avoid unstable score at endpoints (DMD paper)
            if schedule == "Reflow":
                t = 0.02 + 0.96 * torch.rand(bs, device=device)
            else:
                t = (0.02 + 0.96 * torch.rand(bs, device=device)) * (np.pi / 2)
            x_t = interpolate(x0, eps, t, schedule)

            with torch.no_grad():
                pred_real = teacher(x_t, t, c)
            pred_fake = fake_score_model(x_t, t, c)

            alpha_t, _ = get_alpha_sigma_t(t, schedule)
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            loss_dm, grad_norm_dm = distribution_matching_loss(
                x0, pred_real, pred_fake, alpha_t
            )
            t_clean = torch.zeros(bs, device=device)
            logits_gen = discriminator(x0, t_clean, c)
            loss_gan_gen = F.softplus(-logits_gen).mean()
            loss_gen = loss_dm + float(cfg.lambda_gan) * loss_gan_gen

            opt_gen.zero_grad(set_to_none=True)
            loss_gen.backward()
            grad_norm_gen = torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt_gen.step()

            # ----- C. Update student score function μ_fake for n_score_updates denoising steps, while fixing generator G_theta -----
            loss_denoise = None
            for _ in range(cfg.n_score_updates):
                with torch.no_grad():
                    z_s = torch.randn(bs, 3, 32, 32, device=device)
                    c_s = torch.randint(0, cfg.num_classes, (bs,), device=device, dtype=torch.long)
                    x0_fake = one_step_generate(student, z_s, c_s, t_end, device).detach()
                loss_denoise = denoising_loss_fake(
                    fake_score_model, x0_fake, c_s, schedule, device,
                    sigma_data=getattr(cfg, "sigma_data", 0.5),
                )
                opt_score.zero_grad(set_to_none=True)
                loss_denoise.backward()
                torch.nn.utils.clip_grad_norm_(fake_score_model.parameters(), 5.0)
                opt_score.step()

            # ----- D. Update Discriminator step -----
            for _ in range(cfg.n_disc_updates):
                try:
                    real_batch = next(real_iter)
                except StopIteration:
                    real_iter = iter(dataloader)
                    real_batch = next(real_iter)
                x_real = normer.norm(real_batch[0].to(device))
                c_real = real_batch[1].to(device)

                with torch.no_grad():
                    z_d = torch.randn(bs, 3, 32, 32, device=device)
                    c_d = torch.randint(0, cfg.num_classes, (bs,), device=device, dtype=torch.long)
                    x_fake = one_step_generate(student, z_d, c_d, t_end, device)

                for p in student.parameters():
                    p.requires_grad_(False)
                real_pred = discriminator(x_real, t_clean, c_real)
                fake_pred = discriminator(x_fake.detach(), t_clean, c_d)
                loss_disc = softplus_d_loss(real_pred, fake_pred)

                opt_disc.zero_grad(set_to_none=True)
                loss_disc.backward()
                opt_disc.step()
                for p in student.parameters():
                    p.requires_grad_(True)

            

            global_step += 1

            if global_step % cfg.log_step == 0:
                current_time = time.asctime(time.localtime(time.time()))
                log_message = (
                    f"{current_time}\n"
                    f"step={global_step} L_dm={loss_dm.item():.6f} "
                    f"L_gan_gen={loss_gan_gen.item():.6f} L_disc={loss_disc.item():.6f} "
                    f"L_gen={loss_gen.item():.6f}\n"
                )
                with open(log_path, "a") as f:
                    f.write(log_message)

            if wandb_run is not None and global_step % wandb_log_step == 0:
                with torch.no_grad():
                    x0_mean = float(x0.detach().mean().item())
                    x0_std = float(x0.detach().std().item())
                metrics = {
                    "global_step": global_step,
                    "train/loss_dm": float(loss_dm.item()),
                    "train/loss_gan_gen": float(loss_gan_gen.item()),
                    "train/loss_disc": float(loss_disc.item()),
                    "train/loss_gen_total": float(loss_gen.item()),
                    "train/grad_norm_gen": float(grad_norm_gen.item()) if torch.is_tensor(grad_norm_gen) else float(grad_norm_gen),
                    "train/grad_norm_dm": grad_norm_dm,
                    "train/x0_mean": x0_mean,
                    "train/x0_std": x0_std,
                    "train/lr_gen": float(opt_gen.param_groups[0]["lr"]),
                    "train/lr_disc": float(opt_disc.param_groups[0]["lr"]),
                }
                if loss_denoise is not None:
                    metrics["train/loss_denoise"] = float(loss_denoise.item())
                    metrics["train/lr_score"] = float(opt_score.param_groups[0]["lr"])
                wandb.log(metrics)

            if global_step % cfg.sample_step == 0:
                student.eval()
                with torch.no_grad():
                    z_vis = torch.randn(cfg.num_classes, 3, 32, 32, device=device)
                    c_vis = torch.arange(cfg.num_classes, device=device)
                    x_vis = one_step_generate(student, z_vis, c_vis, t_end, device)
                    x_vis = normer.unnorm(x_vis)
                log_img = make_grid(x_vis, nrow=10)
                save_image(log_img, os.path.join(images_dir, f"step_{global_step}.png"))
                student.train()

            if global_step % cfg.ckpt_step == 0:
                ckpt_path = os.path.join(ckpt_dir, f"student_step_{global_step}.pt")
                torch.save(
                    {
                        "model_config": cfg.model_config,
                        "flow_schedule": schedule,
                        "state_dict": student.state_dict(),
                        "global_step": global_step,
                    },
                    ckpt_path,
                )
                _remove_old_checkpoints(ckpt_dir, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            x0_mean_step = float(x0.detach().mean().item())
            x0_std_step = float(x0.detach().std().item())
            pbar.set_postfix(
                dm=f"{loss_dm.item():.4f}",
                gan=f"{loss_gan_gen.item():.4f}",
                disc=f"{loss_disc.item():.4f}",
                gdm=f"{grad_norm_dm:.2f}",
                x0m=f"{x0_mean_step:.3f}",
                x0s=f"{x0_std_step:.3f}",
            )

    ckpt_path = os.path.join(ckpt_dir, f"student_step_{global_step}.pt")
    torch.save(
        {
            "model_config": cfg.model_config,
            "flow_schedule": schedule,
            "state_dict": student.state_dict(),
            "global_step": global_step,
        },
        ckpt_path,
    )
    _remove_old_checkpoints(ckpt_dir, ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")
    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()