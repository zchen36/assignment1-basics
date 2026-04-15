from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F
from llm.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 / 3
        if d_ff % 64 != 0:
            d_ff = ((d_ff + 63) // 64) * 64  # round up to multiple of 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # gate
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # down
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # up

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)
