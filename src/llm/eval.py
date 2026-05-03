import torch
from torch import nn
import numpy.typing as npt
from .cross_entropy import cross_entropy
from .data_loader import get_batch


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    valid_data: npt.NDArray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    vocab_size: int,
    device: str,
) -> float:
    model.eval()

    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(
            dataset=valid_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(x)
        loss = cross_entropy(
            inputs=logits.reshape(-1, vocab_size), targets=y.reshape(-1)
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)
