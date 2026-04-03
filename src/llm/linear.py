from jaxtyping import Float
import torch
from torch import Tensor, nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((d_out, d_in), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
