from jaxtyping import Float, Bool, Int
from torch import Tensor, nn
import torch
from einops import einsum, rearrange
from llm.softmax import softmax
from llm.RoPE import RotaryPositionalEmbedding
from llm.RMSNorm import RMSNorm
from llm.SwiGLU import SwiGLU
from llm.embedding import Embedding
from llm.linear import Linear
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def dot_product_attention(
    queries: Float[Tensor, "batch_size ... q_len d_k"],
    keys: Float[Tensor, "batch_size ... k_len d_k"],
    values: Float[Tensor, "batch_size ... k_len d_v"],
    mask: Bool[Tensor, "... q_len k_len"] = None,
):
    d_k = keys.shape[-1]

    dot_product = (
        einsum(
            queries,
            keys,
            "batch_size ... q_len d_k, batch_size ... k_len d_k -> batch_size ... q_len k_len",
        )
        / d_k**0.5
    )

    if mask is not None:
        dot_product = dot_product.masked_fill(~mask, float("-inf"))

    return einsum(
        softmax(dot_product, dim=-1),
        values,
        "batch_size ... q_len k_len, batch_size ... k_len d_v -> batch_size ... q_len d_v",
    )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_weight: Float[Tensor, "d_model d_k_sum"] = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.k_weight: Float[Tensor, "d_model d_q_sum"] = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.v_weight: Float[Tensor, "d_model d_v_sum"] = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.o_weight: Float[Tensor, "d_model d_v_sum"] = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        mask: Bool[Tensor, "seq_len seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        seq_len = x.shape[-2]

        w_qkv: Float[Tensor, "d_model d_model_3"] = torch.cat(
            [self.q_weight, self.k_weight, self.v_weight], dim=1
        )

        # logging.info(f"x shape: {x.shape}")
        # logging.info(f"q_weight shape: {self.q_weight.shape}")
        # logging.info(f"k_weight shape: {self.k_weight.shape}")
        # logging.info(f"v_weight shape: {self.v_weight.shape}")
        # logging.info(f"w_qkv shape: {w_qkv.shape}")

        xw_qkv = einsum(
            x,
            w_qkv,
            "... seq_len d_model, d_model d_model_3 -> ... seq_len d_model_3",
        )

        qkv = rearrange(xw_qkv, "... (three d_model) -> three ... d_model", three=3)

        q: Float[Tensor, "... seq_len d_k_sum"]
        k: Float[Tensor, "... seq_len d_k_sum"]
        v: Float[Tensor, "... seq_len d_v_sum"]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = rearrange(
            q,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        k = rearrange(
            k,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        v = rearrange(
            v,
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
        )

        if mask is None:
            mask: Bool[Tensor, "... seq_len seq_len"] = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            )

        multi_head_attention: Float[Tensor, "... h seq_len d_v"] = (
            dot_product_attention(q, k, v, mask)
        )

        concat_attention: Float[Tensor, "... seq_len d_v_sum"] = rearrange(
            multi_head_attention, "... h seq_len d_v -> ... seq_len (h d_v)"
        )

        return einsum(
            concat_attention,
            self.o_weight,
            "... d_v_sum, ... d_model d_v_sum -> ... d_model",
        )


class MultiHeadAttentionWithRope(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model({d_model}) must be divisible by num_heads({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_weight: Float[Tensor, "d_model d_k_sum"] = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.k_weight: Float[Tensor, "d_model d_q_sum"] = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.v_weight: Float[Tensor, "d_model d_v_sum"] = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )
        self.o_weight: Float[Tensor, "d_model d_v_sum"] = nn.Parameter(
            torch.empty((d_model, d_model), device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.q_weight, mean=0.0, std=0.02, a=-3, b=3)
        nn.init.trunc_normal_(self.k_weight, mean=0.0, std=0.02, a=-3, b=3)
        nn.init.trunc_normal_(self.v_weight, mean=0.0, std=0.02, a=-3, b=3)
        nn.init.trunc_normal_(self.o_weight, mean=0.0, std=0.02, a=-3, b=3)

        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        mask: Bool[Tensor, "seq_len seq_len"] | None = None,
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        seq_len = x.shape[-2]

        w_qkv: Float[Tensor, "d_model d_model_3"] = torch.cat(
            [self.q_weight, self.k_weight, self.v_weight], dim=1
        )

        # logging.info(f"x shape: {x.shape}")
        # logging.info(f"q_weight shape: {self.q_weight.shape}")
        # logging.info(f"k_weight shape: {self.k_weight.shape}")
        # logging.info(f"v_weight shape: {self.v_weight.shape}")
        # logging.info(f"w_qkv shape: {w_qkv.shape}")

        xw_qkv = einsum(
            x,
            w_qkv,
            "... seq_len d_model, d_model d_model_3 -> ... seq_len d_model_3",
        )

        qkv = rearrange(xw_qkv, "... (three d_model) -> three ... d_model", three=3)

        q: Float[Tensor, "... seq_len d_k_sum"]
        k: Float[Tensor, "... seq_len d_k_sum"]
        v: Float[Tensor, "... seq_len d_v_sum"]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = rearrange(
            q,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        k = rearrange(
            k,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        v = rearrange(
            v,
            "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
            num_heads=self.num_heads,
        )

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        if mask is None:
            mask: Bool[Tensor, "... seq_len seq_len"] = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            )

        multi_head_attention: Float[Tensor, "... h seq_len d_v"] = (
            dot_product_attention(q, k, v, mask)
        )

        concat_attention: Float[Tensor, "... seq_len d_v_sum"] = rearrange(
            multi_head_attention, "... h seq_len d_v -> ... seq_len (h d_v)"
        )

        return einsum(
            concat_attention,
            self.o_weight,
            "... d_v_sum, ... d_model d_v_sum -> ... d_model",
        )


class PreNormTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        d_ff: int,
        theta: float,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.rms_norm_attn = RMSNorm(d_model, eps, device, dtype)
        self.rms_norm_ffn = RMSNorm(d_model, eps, device, dtype)
        self.multi_head_attention = MultiHeadAttentionWithRope(
            d_model, num_heads, max_seq_len, theta, device, dtype
        )
        self.swi_glu = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ):
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len)

        if x.shape[-2] != token_positions.shape[-1]:
            raise ValueError(
                f"seq_len in x(shape: {x.shape}) and token_positions(shape: {token_positions.shape}) must be the same"
            )

        residual = x
        x_norm = self.rms_norm_attn(x)
        att_out = self.multi_head_attention(x_norm, token_positions=token_positions)
        x = residual + att_out

        residual = x
        x_norm = self.rms_norm_ffn(x)
        ffn_out = self.swi_glu(x_norm)
        x = residual + ffn_out
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        # context_length: int, same as max_seq_len?
        num_layers: int,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        d_ff: int,
        theta: float,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embedding = Embedding(
            vocab_size=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )

        self.attn_layers = nn.ModuleList(
            [
                PreNormTransformer(
                    d_model=d_model,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    d_ff=d_ff,
                    theta=theta,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.output_embedding = Linear(
            d_in=d_model, d_out=vocab_size, device=device, dtype=dtype
        )

    def forward(self, x: Int[Tensor, "batch seq_len"]):
        x = self.embedding(x)
        for layer in self.attn_layers:
            x = layer(x)

        return self.output_embedding(self.norm(x))
        # return softmax(x, -1)


if __name__ == "__main__":
    multi_head_attention = MultiHeadAttention(512, 8)
    x = torch.randn(1, 10, 512)
    print(multi_head_attention(x).shape)

    B, H, T, D = 2, 4, 6, 8
    multi_head_attention_with_rope = MultiHeadAttentionWithRope(D, H, T, 10000)
    x = torch.randn(B, T, D)
    token_positions = torch.arange(T).expand(B, H, T)

    y = multi_head_attention_with_rope(x, token_positions=token_positions)
