from models.dit import MFDiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
import argparse
import glob
import time
import os

from config import load_config


def _remove_old_checkpoints(ckpt_dir, keep_path, pattern="step_*.pt"):
    """Remove all checkpoints matching pattern except keep_path."""
    for f in glob.glob(os.path.join(ckpt_dir, pattern)):
        if os.path.abspath(f) != os.path.abspath(keep_path):
            try:
                os.remove(f)
            except OSError:
                pass


def get_dataset(cfg):
    """Build dataset and dataloader from config."""
    if cfg.transform == "mnist":
        transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    elif cfg.transform == "cifar10":
        transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
    else:
        raise ValueError(f"Unknown transform: {cfg.transform}")

    if cfg.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(
            root=cfg.root,
            train=True,
            download=True,
            transform=transform,
        )
    elif cfg.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=cfg.root,
            train=True,
            download=True,
            transform=transform,
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
    train_dataloader = cycle(train_dataloader)
    return dataset, train_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset (mnist|cifar10)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output root for images/ and checkpoints/ (default: MF/ when run from Train/)")
    args = parser.parse_args()

    cfg = load_config(path=args.config, dataset_override=args.dataset)

    # Output dir: images/ and checkpoints/ under MF/ (parent of Train/)
    output_dir = args.output_dir or os.environ.get("MF_OUTPUT_DIR", "..")
    images_dir = os.path.join(output_dir, "images")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16")

    _, train_dataloader = get_dataset(cfg)

    model = MFDiT(**cfg.model_config).to(accelerator.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    meanflow = MeanFlow(**cfg.meanflow_config)

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

    global_step = start_step
    losses = 0.0
    mse_losses = 0.0

    with tqdm(range(start_step, cfg.n_steps), 
                   initial=start_step, 
                   total=cfg.n_steps, 
                   dynamic_ncols=True,
                   disable=not accelerator.is_main_process) as pbar:
        pbar.set_description("Training")
        model.train()
        for step in pbar:
            data = next(train_dataloader)
            x = data[0].to(accelerator.device)
            c = data[1].to(accelerator.device)

            loss, mse_val = meanflow.loss(model, x, c)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()
            mse_losses += mse_val.item()

            if accelerator.is_main_process:
                if global_step % cfg.log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    batch_info = f"Global Step: {global_step}"
                    loss_info = (
                        f"Loss: {losses / cfg.log_step:.6f}    "
                        f"MSE_Loss: {mse_losses / cfg.log_step:.6f}"
                    )
                    lr = optimizer.param_groups[0]["lr"]
                    lr_info = f"Learning Rate: {lr:.6f}"
                    log_message = f"{current_time}\n{batch_info}    {loss_info}    {lr_info}\n"
                    with open("log.txt", mode="a") as f:
                        f.write(log_message)
                    losses = 0.0
                    mse_losses = 0.0

            if global_step % cfg.sample_step == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, "module") else model
                    z = meanflow.sample_each_class(
                        model_module, 1,
                        num_steps=cfg.num_steps,
                        device=accelerator.device,
                        use=cfg.sample_use,
                    )
                    log_img = make_grid(z, nrow=10)
                    img_save_path = os.path.join(images_dir, f"step_{global_step}.png")
                    save_image(log_img, img_save_path)
                accelerator.wait_for_everyone()
                model.train()

            if global_step % cfg.ckpt_step == 0 and accelerator.is_main_process:
                model_module = model.module if hasattr(model, "module") else model
                ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                ckpt = {
                    "model_config": cfg.model_config,
                    "meanflow_config": cfg.meanflow_config,
                    "state_dict": model_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                }
                torch.save(ckpt, ckpt_path)
                _remove_old_checkpoints(ckpt_dir, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    if accelerator.is_main_process:
        model_module = model.module if hasattr(model, "module") else model
        ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
        ckpt = {
            "model_config": cfg.model_config,
            "meanflow_config": cfg.meanflow_config,
            "state_dict": model_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }
        torch.save(ckpt, ckpt_path)
        _remove_old_checkpoints(ckpt_dir, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
