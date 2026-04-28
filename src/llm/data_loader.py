import torch
import numpy.typing as npt
import numpy as np


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    N = len(dataset)

    starts = np.random.randint(0, N - context_length, size=batch_size)

    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])

    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)

    return (x, y)
