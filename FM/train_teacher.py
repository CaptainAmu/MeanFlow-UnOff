"""
Flow matching teacher training: clean prediction x0_hat(x_t, t, y).
Supports Reflow and Trig flow schedules.
"""
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from accelerate import Accelerator
import argparse
import glob
import time
import os
import numpy as np

from models.dit import FlowMatchDiT


def _remove_old_checkpoints(ckpt_dir, keep_path, pattern="step_*.pt"):
    """Remove all checkpoints matching pattern except keep_path."""
    for f in glob.glob(os.path.join(ckpt_dir, pattern)):
        if os.path.abspath(f) != os.path.abspath(keep_path):
            try:
                os.remove(f)
            except OSError:
                pass


from config import load_config
from utils.flow_schedule import interpolate, get_t_end, velocity_from_x0


class Normalizer:
    def __init__(self, mode="minmax", mean=None, std=None):
        self.mode = mode
        if mode == "mean_std":
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        mode, mean, std = config
        return cls(mode, mean, std)

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
    if cfg.transform == "mnist":
        transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    elif cfg.transform == "cifar10":
        transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
    else:
        raise ValueError(f"Unknown transform: {cfg.transform}")

    if cfg.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(
            root=cfg.root, train=True, download=True, transform=transform
        )
    elif cfg.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=cfg.root, train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )
    return dataset, cycle(train_dataloader)


def sample_t(batch_size, device, schedule):
    """Sample t: Reflow [0,1], Trig [0, pi/2]."""
    if schedule == "Reflow":
        return torch.rand(batch_size, device=device)
    elif schedule == "Trig":
        return torch.rand(batch_size, device=device) * (np.pi / 2)
    raise ValueError(f"Unknown schedule: {schedule}")


@torch.no_grad()
def sample_each_class(
    model,
    normer,
    schedule,
    n_per_class=1,
    device="cuda",
    euler_steps=1,
):
    """
    Sample one image per class using Euler ODE integration.
    euler_steps=1: 1-step clean prediction at t_end (fast).
    euler_steps>1: multi-step Euler integration from noise (t_end) to data (t=0).
    """
    model.eval()
    t_end = get_t_end(schedule)
    c = torch.arange(model.num_classes, device=device).repeat(n_per_class)
    z = torch.randn(
        c.shape[0], model.out_channels, model.input_size, model.input_size, device=device
    )
    t_vals = torch.linspace(t_end, 0.0, euler_steps + 1, device=device)
    for i in range(euler_steps):
        t_i = t_vals[i]
        t_next = t_vals[i + 1]
        dt = t_i - t_next  # positive when going from t_end to 0
        t_batch = torch.full((z.shape[0],), t_i, device=device, dtype=z.dtype)
        x0_hat = model(z, t_batch, c)
        v = velocity_from_x0(z, x0_hat, t_batch, schedule)
        z = z - dt * v
    return normer.unnorm(z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_teacher.yml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(path=args.config, dataset_override=args.dataset)
    output_dir = args.output_dir or os.environ.get("FM_OUTPUT_DIR", ".")
    images_dir = os.path.join(output_dir, "train_images")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16")
    _, train_dataloader = get_dataset(cfg)

    model = FlowMatchDiT(**cfg.model_config).to(accelerator.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    normer = Normalizer.from_list(cfg.normalizer)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model_module = model.module if hasattr(model, "module") else model
        model_module.load_state_dict(ckpt["state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["global_step"])
        if accelerator.is_main_process:
            print(f"Resumed from {args.resume} at step {start_step}")

    schedule = cfg.flow_schedule
    global_step = start_step
    losses = 0.0

    with tqdm(
        range(start_step, cfg.n_steps),
        initial=start_step,
        total=cfg.n_steps,
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    ) as pbar:
        pbar.set_description("FM Teacher")
        model.train()
        for step in pbar:
            data = next(train_dataloader)
            x = data[0].to(accelerator.device)
            c = data[1].to(accelerator.device)

            x = normer.norm(x)
            eps = torch.randn_like(x, device=x.device)
            t = sample_t(x.shape[0], x.device, schedule)

            x_t = interpolate(x, eps, t, schedule)
            x0_hat = model(x_t, t, c)
            loss = torch.nn.functional.mse_loss(x0_hat, x)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()

            if accelerator.is_main_process:
                if global_step % cfg.log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    log_message = (
                        f"{current_time}\n"
                        f"Global Step: {global_step}    "
                        f"Loss: {losses / cfg.log_step:.6f}    "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
                    )
                    log_path = os.path.join(output_dir, "log.txt")
                    with open(log_path, "a") as f:
                        f.write(log_message)
                    losses = 0.0

            if global_step % cfg.sample_step == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, "module") else model
                    z = sample_each_class(
                        model_module, normer, schedule, 1, accelerator.device, cfg.euler_steps
                    )
                    log_img = make_grid(z, nrow=10)
                    save_image(log_img, os.path.join(images_dir, f"step_{global_step}.png"))
                accelerator.wait_for_everyone()
                model.train()

            if global_step % cfg.ckpt_step == 0 and accelerator.is_main_process:
                model_module = model.module if hasattr(model, "module") else model
                ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                torch.save(
                    {
                        "model_config": cfg.model_config,
                        "flow_schedule": schedule,
                        "state_dict": model_module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                    },
                    ckpt_path,
                )
                _remove_old_checkpoints(ckpt_dir, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    if accelerator.is_main_process:
        model_module = model.module if hasattr(model, "module") else model
        ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
        torch.save(
            {
                "model_config": cfg.model_config,
                "flow_schedule": schedule,
                "state_dict": model_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            ckpt_path,
        )
        _remove_old_checkpoints(ckpt_dir, ckpt_path)
        print(f"Saved final checkpoint to {ckpt_path}")
