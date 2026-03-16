"""
DMD Student Score Function.

The student score function is implemented via clean prediction: the network predicts
μ_θ(x_t, t, y) (clean image), and the score is derived as

    s_θ(x_t, t) = -(x_t - α_t μ_θ(x_t, t, y)) / σ_t²

This follows the flow matching formulation where x_t = α_t x_0 + σ_t ε.

In DMD distillation, we alternate:
  1. Update one-step generator G_θ (distribution matching + GAN)
  2. Update student score via standard score matching (MSE on score or equivalent)

The student score network uses the same DiT architecture as the teacher/student
(FlowMatchDiT), since both predict the clean image μ from (x_t, t, y).
"""
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit import FlowMatchDiT
from utils.flow_schedule import get_alpha_sigma_t, interpolate


def score_from_clean_pred(
    x_t: torch.Tensor,
    mu: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t_sq: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute score s(x_t, t) from clean prediction μ.

    s(x_t, t) = -(x_t - α_t μ) / σ_t²

    Args:
        x_t: (B, C, H, W) noisy input
        mu: (B, C, H, W) clean prediction μ_θ(x_t, t, y)
        alpha_t: (B,) or (B,1,1,1) flow coefficient
        sigma_t_sq: (B,) or (B,1,1,1) σ_t²
        eps: small constant for numerical stability

    Returns:
        s: (B, C, H, W) score
    """
    if alpha_t.dim() == 1:
        alpha_t = alpha_t.view(-1, 1, 1, 1)
    if sigma_t_sq.dim() == 1:
        sigma_t_sq = sigma_t_sq.view(-1, 1, 1, 1)
    sigma_t_sq = sigma_t_sq.clamp(min=eps)
    return -(x_t - alpha_t * mu) / sigma_t_sq


class StudentScoreNetwork(nn.Module):
    """
    Student score function implemented via clean prediction (DiT backbone).

    Wraps a FlowMatchDiT that predicts μ(x_t, t, y). The score is
    s(x_t, t) = -(x_t - α_t μ) / σ_t².

    Used in DMD for the score-matching update step.
    """

    def __init__(
        self,
        backbone: Optional[FlowMatchDiT] = None,
        **dit_kwargs,
    ):
        """
        Args:
            backbone: Existing FlowMatchDiT to wrap. If None, creates one from dit_kwargs.
            **dit_kwargs: Passed to FlowMatchDiT if backbone is None.
        """
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = FlowMatchDiT(**dit_kwargs)

    def predict_clean(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict clean image μ_θ(x_t, t, y)."""
        return self.backbone(x_t, t, y)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        schedule: Literal["Reflow", "Trig"],
    ) -> torch.Tensor:
        """
        Compute student score s_θ(x_t, t).

        Args:
            x_t: (B, C, H, W) noisy input
            t: (B,) timestep
            y: (B,) class labels
            schedule: "Reflow" or "Trig"

        Returns:
            s: (B, C, H, W) score s_θ(x_t, t) = -(x_t - α_t μ) / σ_t²
        """
        mu = self.predict_clean(x_t, t, y)
        alpha_t, sigma_t = get_alpha_sigma_t(t, schedule)
        sigma_t_sq = sigma_t ** 2
        return score_from_clean_pred(x_t, mu, alpha_t, sigma_t_sq)


def score_matching_loss(
    score_net: StudentScoreNetwork,
    x0: torch.Tensor,
    labels: torch.Tensor,
    schedule: Literal["Reflow", "Trig"],
    device: torch.device,
) -> torch.Tensor:
    """
    Standard score matching loss for the student score network.

    Sample x_t = α_t x0 + σ_t ε, target score s_target = -(x_t - α_t x0) / σ_t².
    Loss = MSE(s_θ(x_t, t), s_target).

    Args:
        score_net: StudentScoreNetwork
        x0: (B, C, H, W) clean samples (e.g. from G or real data)
        labels: (B,) class labels
        schedule: "Reflow" or "Trig"
        device: torch device

    Returns:
        loss: scalar
    """
    batch_size = x0.shape[0]
    eps = torch.randn_like(x0, device=device)
    if schedule == "Reflow":
        t = torch.rand(batch_size, device=device)
    else:
        t = torch.rand(batch_size, device=device) * (3.141592653589793 / 2)
    x_t = interpolate(x0, eps, t, schedule)

    s_pred = score_net(x_t, t, labels, schedule)
    alpha_t, sigma_t = get_alpha_sigma_t(t, schedule)
    alpha_t = alpha_t.view(-1, 1, 1, 1)
    sigma_t_sq = (sigma_t ** 2).view(-1, 1, 1, 1).clamp(min=1e-8)
    s_target = -(x_t - alpha_t * x0) / sigma_t_sq

    return F.mse_loss(s_pred, s_target, reduction="mean")
