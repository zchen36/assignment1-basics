from jaxtyping import Float, Int
import torch
from torch import Tensor, nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # Compute RMS: sqrt(mean(x^2) + eps)
        # We take the mean over the last dimension (d_model)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # return self.scale * (x * rms)
        return einsum(self.scale, x * rms, "d_model, ... d_model -> ... d_model")
