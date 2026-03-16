"""
Flow schedules: Reflow and Trig.
Used by FM teacher training and DMD student distillation.
"""
from typing import Literal
import torch

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


def get_t_end(schedule: Literal["Reflow", "Trig"]) -> float:
    """Time at noise endpoint: Reflow t=1, Trig t=pi/2."""
    if schedule == "Reflow":
        return 1.0
    elif schedule == "Trig":
        return 3.141592653589793 / 2
    else:
        raise ValueError(f"Unknown flow_schedule: {schedule}")
