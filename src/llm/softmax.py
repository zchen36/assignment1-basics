from jaxtyping import Float, Int
from torch import Tensor, nn
import torch


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    max_value = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_value
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x
