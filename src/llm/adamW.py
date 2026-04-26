from jaxtyping import Float, Int
from torch import Tensor
import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    ):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["first_moment"] = torch.zeros_like(p)
                    state["second_moment"] = torch.zeros_like(p)

                t = state["t"] + 1
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]

                grad = p.grad.data

                first_moment.mul_(beta1).add_(grad, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                lr_adjusted = lr * (math.sqrt(1 - beta2**t)) / (1 - beta1**t)

                denom = second_moment.sqrt().add_(eps)
                p.data.addcdiv_(first_moment, denom, value=-lr_adjusted)
                p.data.sub_(p.data, alpha=lr * weight_decay)

                state["t"] = t


def _optimize(opt_class, steps: int = 1000) -> Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(steps):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()

    return model.weight.detach()


def main() -> None:
    learned_weights = _optimize(AdamW)
    reference_weights = _optimize(torch.optim.AdamW)

    print("Learned weights:")
    print(learned_weights)
    print()
    print("PyTorch AdamW reference weights:")
    print(reference_weights)
    print()
    print(
        "Close to PyTorch reference:",
        torch.allclose(learned_weights, reference_weights, atol=1e-4),
    )


if __name__ == "__main__":
    main()
