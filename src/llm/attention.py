from jaxtyping import Float, Bool
from torch import Tensor
from einops import einsum
from llm.softmax import softmax


def dot_product_attention(
    queries: Float[Tensor, "batch_size ... q_len d_k"],
    keys: Float[Tensor, "batch_size ... k_len d_k"],
    values: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[Tensor, "... q_len k_len"] = None,
):
    d_k = keys.shape[-1]

    dot_product = (
        einsum(queries, keys, "batch_size ... q_len d_k, batch_size ... k_len d_k -> batch_size ... q_len k_len")
        / d_k**0.5
    )

    if mask is not None:
        dot_product = dot_product.masked_fill(~mask, float("-inf"))

    return einsum(
        softmax(dot_product, dim=-1),
        values,
        "batch_size ... q_len k_len, batch_size ... k_len d_v -> batch_size ... q_len d_v",
    )
