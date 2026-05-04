from jaxtyping import Float, Int
from torch import Tensor, nn
import torch


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be divisible by 2")
        if theta <= 0:
            raise ValueError("theta must be greater than 0")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than 0")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # (d_k / 2,)
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.d_k, 2, dtype=torch.float32, device=self.device)
                / self.d_k
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # (max_seq_len,)
        position = torch.arange(
            self.max_seq_len, dtype=torch.float32, device=self.device
        )

        # (max_seq_len, d_k / 2)
        freqs = torch.outer(position, self.inv_freq)

        # (max_seq_len, d_k)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Float[Tensor, "... d_k"]) -> Float[Tensor, "... d_k"]:
        """(x1, x2, ...) -> (-x2, x1, ...)"""

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack([-x_odd, x_even], dim=-1)
        return x_rot.flatten(start_dim=-2)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"x.shape[-1] ({x.shape[-1]}) != self.d_k ({self.d_k})")

        if x.shape[-2] != token_positions.shape[-1]:
            raise ValueError(
                f"seq_len in x(shape: {x.shape}) and token_positions(shape: {token_positions.shape}) must be the same"
            )

        if token_positions.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(
                f"token_positions.dtype ({token_positions.dtype}) must be an integer type"
            )

        # if token_positions.numel() > 0:
        #     pos_min = token_positions.min().item()
        #     pos_max = token_positions.max().item()
        #     if pos_min < 0 or pos_max >= self.max_seq_len:
        #         raise ValueError(f"token_positions contains index out of bounds (min: {pos_min}, max: {pos_max})")

        flat_pos = token_positions.reshape(-1).to(
            device=self.cos_cached.device, dtype=torch.long
        )
        cos = self.cos_cached.index_select(0, flat_pos).reshape(
            *token_positions.shape, self.d_k
        )
        sin = self.sin_cached.index_select(0, flat_pos).reshape(
            *token_positions.shape, self.d_k
        )

        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)

        return x * cos + self._rotate_half(x) * sin


if __name__ == "__main__":
    B, H, T, D = 2, 4, 6, 8

    rope = RotaryPositionalEmbedding(
        theta=10000.0,
        d_k=D,
        max_seq_len=128,
    )

    x = torch.randn(B, H, T, D)
    token_positions = torch.arange(T).expand(B, H, T)

    y = rope(x, token_positions)
    print(y.shape)
