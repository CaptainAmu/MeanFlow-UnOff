"""
Flow schedules: Reflow and Trig.
Used by FM teacher training and DMD student distillation.
"""
from typing import Literal, Tuple
import torch
import numpy as np


def get_alpha_sigma_t(
    t: torch.Tensor,
    schedule: Literal["Reflow", "Trig"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (α_t, σ_t) for flow matching. x_t = α_t*x_0 + σ_t*ε.
    Reflow: α_t = 1-t, σ_t = t
    Trig:   α_t = cos(t), σ_t = sin(t)
    """
    if schedule == "Reflow":
        alpha_t = 1 - t
        sigma_t = t
    elif schedule == "Trig":
        alpha_t = torch.cos(t)
        sigma_t = torch.sin(t)
    else:
        raise ValueError(f"Unknown flow_schedule: {schedule}")
    return alpha_t, sigma_t


def interpolate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    schedule: Literal["Reflow", "Trig"],
) -> torch.Tensor:
    """
    Reflow: x_t = (1-t)x_0 + t x_1, t in [0, 1]
    Trig:   x_t = cos(t)x_0 + sin(t)x_1, t in [0, pi/2]
    """
    if t.dim() == 1:
        t = t.view(-1, 1, 1, 1)
    elif t.dim() == 2:
        t = t.view(-1, 1, 1, 1)
    if schedule == "Reflow":
        return (1 - t) * x0 + t * x1
    elif schedule == "Trig":
        return torch.cos(t) * x0 + torch.sin(t) * x1
    else:
        raise ValueError(f"Unknown flow_schedule: {schedule}")


def get_denoising_weight(
    t: torch.Tensor,
    schedule: Literal["Reflow", "Trig"],
    sigma_data: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    EDM-style weight for denoising loss: w = min(base_weight, 40 * sigma_data^-2).
    Reflow: base = (1-t)^(-2) * sigma_data^(-2) + t^(-2)
    Trig:   base = (cos(t))^(-2) * sigma_data^(-2) + (sin(t))^(-2)
    Returns (B,) per-sample weights.
    """
    cap = 40.0 / (sigma_data**2)
    if schedule == "Reflow":
        t_safe = t.clamp(min=eps, max=1.0 - eps)
        w = (1.0 - t_safe) ** (-2) / (sigma_data**2) + (t_safe**2) ** (-1)
    elif schedule == "Trig":
        cost = torch.cos(t).clamp(min=eps, max=1.0)
        sint = torch.sin(t).clamp(min=eps, max=1.0)
        w = (cost**2) ** (-1) / (sigma_data**2) + (sint**2) ** (-1)
    else:
        raise ValueError(f"Unknown flow_schedule: {schedule}")
    return torch.clamp(w, max=cap)


def get_t_end(schedule: Literal["Reflow", "Trig"]) -> float:
    """Time at noise endpoint: Reflow t=1, Trig t=pi/2."""
    if schedule == "Reflow":
        return 1.0
    elif schedule == "Trig":
        return np.pi / 2
    else:
        raise ValueError(f"Unknown flow_schedule: {schedule}")


def velocity_from_x0(
    x_t: torch.Tensor,
    x0_hat: torch.Tensor,
    t: torch.Tensor,
    schedule: Literal["Reflow", "Trig"],
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute velocity v = dx_t/dt from clean prediction x0_hat.
    Reflow: x_t = (1-t)x_0 + t*x_1, v = (x_t - x0_hat)/t
    Trig:   x_t = cos(t)x_0 + sin(t)*x_1, v = (cos(t)*x_t - x0_hat)/sin(t)
    """
    if t.dim() == 0:
        t = t.unsqueeze(0).expand(x_t.shape[0])
    t_ = t.view(-1, 1, 1, 1) if x_t.dim() == 4 else t.view(-1, 1)
    t_ = t_.to(x_t.dtype)
    if schedule == "Reflow":
        t_safe = t_.clamp(min=eps)
        return (x_t - x0_hat) / t_safe
    elif schedule == "Trig":
        sin_t = torch.sin(t_).clamp(min=eps)
        return (torch.cos(t_) * x_t - x0_hat) / sin_t
    else:
        raise ValueError(f"Unknown flow_schedule: {schedule}")
