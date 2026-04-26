import math


def lr_cos_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters

    if it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1
            + math.cos(
                (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
            )
        ) * (max_learning_rate - min_learning_rate)

    return min_learning_rate
