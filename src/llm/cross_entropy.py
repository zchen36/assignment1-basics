from jaxtyping import Float, Int
from torch import Tensor, nn
import torch


def cross_entropy(
    inputs: Float[Tensor, "batch vocab_size"], targets: Int[Tensor, "batch"]
) -> Float[Tensor, ""]:
    # mean -log(e^x / sum(e^x)), mean -x + log(sum(e^x))

    max_value = torch.max(inputs, dim=1, keepdim=True).values
    inputs_stable = inputs - max_value

    log_sum_exp = torch.log(torch.exp(inputs_stable).sum(dim=1))

    correct_logits = inputs_stable[torch.arange(inputs.shape[0]), targets]

    losses = -correct_logits + log_sum_exp
    return losses.mean()
