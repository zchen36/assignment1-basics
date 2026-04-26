from collections.abc import Iterable
import torch
import math


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_norm = math.sqrt(
        sum(p.grad.norm(2) ** 2 for p in parameters if p.grad is not None)
    )
    if total_norm <= max_l2_norm:
        return

    scale = max_l2_norm / (total_norm + 1e-6)
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)
