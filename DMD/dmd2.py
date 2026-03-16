"""
DMD2 components: Discriminator, distribution matching loss, denoising loss.
Flow matching formulation: x0_hat(x_t, t, y).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal

from utils.flow_schedule import interpolate, get_t_end, get_denoising_weight


def distribution_matching_loss(
    x0: torch.Tensor,
    pred_real: torch.Tensor,
    pred_fake: torch.Tensor,
    alpha_t: torch.Tensor,
    eps: float = 1e-8,
) -> tuple:
    """
    DMD2 distribution matching gradient injection (DMD paper formula).
    p_real = x0 - pred_real, p_fake = x0 - pred_fake
    grad = α_t * (p_real - p_fake) / (mean|p_real| + eps)
    Returns (loss, grad_norm) for monitoring.
    """
    with torch.no_grad():
        p_real = x0 - pred_real
        p_fake = x0 - pred_fake
        weight_factor = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) + eps
        grad = alpha_t * (p_real - p_fake) / weight_factor
        grad = torch.nan_to_num(grad)
        grad_norm = float(torch.norm(grad).item())
    loss = 0.5 * F.mse_loss(x0, (x0 - grad).detach(), reduction="mean")
    return loss, grad_norm


def denoising_loss_fake(
    score_model: nn.Module,
    x0_fake: torch.Tensor,
    labels: torch.Tensor,
    schedule: Literal["Reflow", "Trig"],
    device: torch.device,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    Train score model (μ_fake) on fake samples: x0_fake = G(z,c), x_t = interpolate(x0_fake, eps, t).
    Loss = E[w(t) * MSE(score_model(x_t, t, c), x0_fake)] with EDM-style weights.
    x0_fake should be detached (no grad to generator).
    """
    batch_size = x0_fake.shape[0]
    eps = torch.randn_like(x0_fake, device=device)
    if schedule == "Reflow":
        t = torch.rand(batch_size, device=device)
        t = t.clamp(min=0.01, max=0.99)
    else:
        t = torch.rand(batch_size, device=device) 
        t = t.clamp(min=0.01, max=0.99)
        t = t * (np.pi / 2)
    x_t = interpolate(x0_fake, eps, t, schedule)
    x0_hat = score_model(x_t, t, labels)
    weights = get_denoising_weight(t, schedule, sigma_data)
    weights = weights.view(-1, 1, 1, 1)
    loss = (weights * (x0_hat - x0_fake) ** 2).mean()
    return loss


def softplus_d_loss(real_pred, fake_pred):
    """Softplus loss for discriminator (BCE-style)."""
    return F.softplus(-real_pred).mean() + F.softplus(fake_pred).mean()


class BottleneckDiscriminator(nn.Module):
    """
    Discriminator using student backbone bottleneck + cls_pred_branch (DMD2-style).
    Shares student backbone; only cls_pred_branch is trained from scratch for disc updates.
    """

    def __init__(self, student: nn.Module, dim: int, patch_size: int = 2):
        super().__init__()
        self.student = student
        self.dim = dim
        self.patch_size = patch_size
        self.num_patches_per_side = 32 // patch_size  # 16 for 32x32
        self.cls_pred_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(32, dim // 4), num_channels=dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=4, stride=4, padding=0),
            nn.GroupNorm(num_groups=min(32, dim // 4), num_channels=dim),
            nn.SiLU(),
            nn.Conv2d(dim, 1, kernel_size=1, stride=1, padding=0),
        )
        self.cls_pred_branch.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t, y=None):
        """
        x: (B, C, H, W), t: (B,), y: (B,)
        Returns: (B,) logits
        """
        h = self.student.forward_bottleneck(x, t, y)
        B, N, D = h.shape
        side = int(N**0.5)
        h = h.permute(0, 2, 1).reshape(B, D, side, side)
        logits = self.cls_pred_branch(h).squeeze(dim=[2, 3])
        return logits
