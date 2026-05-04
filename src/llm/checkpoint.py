import torch
from typing import IO, BinaryIO
import os


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    obj = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model_state_dict = obj["model"]

    if any(key.startswith("_orig_mod.") for key in model_state_dict):
        model_state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in model_state_dict.items()
        }

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
