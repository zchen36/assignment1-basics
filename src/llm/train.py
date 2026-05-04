"""
# transformer
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
    )

# lr
def lr_cos_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
)

# gradient clipping: after backward, before taking opt step
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):

get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str)

# loss
cross_entropy(
    inputs: Float[Tensor, "batch vocab_size"], targets: Int[Tensor, "batch"]
) -> Float[Tensor, ""]

# optimizer
class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

# save ckpt
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
)
"""

import numpy as np
import torch
import wandb
from datetime import datetime
from pathlib import Path

from .data_loader import get_batch
from .attention import Transformer
from .cross_entropy import cross_entropy
from .adamW import AdamW
from .gradient_clipping import gradient_clipping
from .lr_schedule import lr_cos_schedule
from .eval import estimate_loss
from .checkpoint import save_checkpoint


def main():
    ROOT = Path(__file__).resolve().parents[2]
    config = {
        # data
        "train_path": ROOT / "data/ts/full_encoding_result/encoded_train.npy",
        "valid_path": ROOT / "data/ts/full_encoding_result/encoded_valid.npy",
        # model
        "vocab_size": 10_000,
        "num_layers": 4,
        "d_model": 512,
        "num_heads": 16,
        "max_seq_len": 256,
        "context_length": 256,
        "d_ff": 1344,
        "theta": 10000.0,
        "eps": 1e-5,
        # optimization
        "batch_size": 32,
        "max_iters": 20_000,
        "max_lr": 5e-4,
        "min_lr": 5e-5,
        "warmup_iters": 500,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "optimizer_eps": 1e-8,
        "grad_clip": 1.0,
        # eval / logging / checkpointing
        "log_interval": 10,
        "eval_interval": 500,
        "eval_iters": 100,
        "ckpt_interval": 1000,
        "ckpt_dir": ROOT / f"data/ckpt/{datetime.now().strftime('%Y_%m_%d_%H_%M')}",
        "wandb_project": "cs336_assignment_1",
    }

    assert config["context_length"] <= config["max_seq_len"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"train on {device}")
    assert device == "mps"

    train_data = np.load(config["train_path"], mmap_mode="r")
    valid_data = np.load(config["valid_path"], mmap_mode="r")

    model = Transformer(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        max_seq_len=config["max_seq_len"],
        d_ff=config["d_ff"],
        theta=config["theta"],
        eps=config["eps"],
        device=device,
    ).to(device)

    model = torch.compile(model, backend="aot_eager")

    optimizer = AdamW(
        params=model.parameters(),
        lr=config["max_lr"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        eps=config["optimizer_eps"],
    )

    ckpt_dir = Path(config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with wandb.init(project=config["wandb_project"], config=config) as run:
        for it in range(config["max_iters"]):
            model.train()

            lr = lr_cos_schedule(
                it=it,
                max_learning_rate=config["max_lr"],
                min_learning_rate=config["min_lr"],
                warmup_iters=config["warmup_iters"],
                cosine_cycle_iters=config["max_iters"],
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(
                dataset=train_data,
                batch_size=config["batch_size"],
                context_length=config["context_length"],
                device=device,
            )

            logits = model(x)  # [batch, seq_len, vocab_size]

            loss = cross_entropy(
                inputs=logits.reshape(-1, config["vocab_size"]),
                targets=y.reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = gradient_clipping(
                parameters=model.parameters(),
                max_l2_norm=config["grad_clip"],
            )

            optimizer.step()

            if it % config["log_interval"] == 0:
                run.log(
                    {"train/loss": loss.item(), "lr": lr, "grad_norm": grad_norm},
                    step=it,
                )

            if it % config["eval_interval"] == 0:
                val_loss = estimate_loss(
                    model=model,
                    valid_data=valid_data,
                    batch_size=config["batch_size"],
                    context_length=config["context_length"],
                    eval_iters=config["eval_iters"],
                    vocab_size=config["vocab_size"],
                    device=device,
                )

                run.log(
                    {"val/loss": val_loss},
                    step=it,
                )

                print(
                    f"iter {it}:"
                    f"train loss {loss.item():.4f}, "
                    f"val loss {val_loss:.4f}, "
                    f"lr {lr:.2e}"
                )

            if it > 0 and it % config["ckpt_interval"] == 0:
                ckpt_path = ckpt_dir / f"ckpt_iter_{it}.pt"

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=it,
                    out=ckpt_path,
                )

                artifact = wandb.Artifact(name=f"ckpt-iter-{it}", type="model")
                artifact.add_file(str(ckpt_path))
                run.log_artifact(artifact)

        final_ckpt_path = ckpt_dir / "ckpt_final.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=config["max_iters"],
            out=final_ckpt_path,
        )


if __name__ == "__main__":
    main()
