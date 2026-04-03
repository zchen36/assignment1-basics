from jaxtyping import Float, Int
import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((vocab_size, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, x: Int[Tensor, "batch sequence_length"]) -> Float[Tensor, "batch sequence_length embedding_dim"]:
        return self.weight[x]


if __name__ == "__main__":
    embedding = Embedding(10, 5)
    print(embedding.weight)

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(embedding(x))
    print("hello")
