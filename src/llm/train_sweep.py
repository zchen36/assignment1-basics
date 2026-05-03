import numpy as np
import torch
import wandb
from datetime import datetime
from pathlib import Path

from llm.data_loader import get_batch
from llm.attention import Transformer
from llm.cross_entropy import cross_entropy
from llm.adamW import AdamW
from llm.gradient_clipping import gradient_clipping
from llm.lr_schedule import lr_cos_schedule
from llm.eval import estimate_loss
from llm.checkpoint import save_checkpoint


def train():
    ROOT = Path(__file__).resolve().parents[2]

    base_config = {
        # data
        "train_path": str(ROOT / "data/ts/full_encoding_result/encoded_train.npy"),
        "valid_path": str(ROOT / "data/ts/full_encoding_result/encoded_valid.npy"),
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
        "max_iters": 10_000,
        "max_lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 500,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "optimizer_eps": 1e-8,
        "grad_clip": 1.0,
        # eval / logging / checkpointing
        "log_interval": 10,
        "eval_interval": 500,
        "eval_iters": 100,
        "ckpt_interval": 1000,
        "save_checkpoints": False,
        # compile
        "compile": False,
        "compile_backend": "aot_eager",
    }

    with wandb.init(
        project="cs336_assignment_1_sweep",
        config=base_config,
    ) as run:
        config = wandb.config

        assert config.context_length <= config.max_seq_len
        assert config.d_model % config.num_heads == 0

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"train on {device}")

        train_data = np.load(config.train_path, mmap_mode="r")
        valid_data = np.load(config.valid_path, mmap_mode="r")

        model = Transformer(
            vocab_size=config.vocab_size,
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            d_ff=config.d_ff,
            theta=config.theta,
            eps=config.eps,
            device=device,
        ).to(device)

        if config.compile:
            model = torch.compile(model, backend=config.compile_backend)

        optimizer = AdamW(
            params=model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.optimizer_eps,
        )

        ckpt_dir = (
            ROOT
            / "data"
            / "ckpt"
            / "sweeps"
            / run.id
            / datetime.now().strftime("%Y_%m_%d_%H_%M")
        )

        if config.save_checkpoints:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        for it in range(config.max_iters):
            model.train()

            lr = lr_cos_schedule(
                it=it,
                max_learning_rate=config.max_lr,
                min_learning_rate=config.min_lr,
                warmup_iters=config.warmup_iters,
                cosine_cycle_iters=config.max_iters,
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(
                dataset=train_data,
                batch_size=config.batch_size,
                context_length=config.context_length,
                device=device,
            )

            logits = model(x)

            loss = cross_entropy(
                inputs=logits.reshape(-1, config.vocab_size),
                targets=y.reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = gradient_clipping(
                parameters=model.parameters(),
                max_l2_norm=config.grad_clip,
            )

            optimizer.step()

            if it % config.log_interval == 0:
                run.log(
                    {
                        "train/loss": loss.item(),
                        "lr": lr,
                        "grad_norm": grad_norm,
                    },
                    step=it,
                )

            if it % config.eval_interval == 0:
                val_loss = estimate_loss(
                    model=model,
                    valid_data=valid_data,
                    batch_size=config.batch_size,
                    context_length=config.context_length,
                    eval_iters=config.eval_iters,
                    vocab_size=config.vocab_size,
                    device=device,
                )

                run.log({"val/loss": val_loss}, step=it)

                print(
                    f"iter {it}: "
                    f"train loss {loss.item():.4f}, "
                    f"val loss {val_loss:.4f}, "
                    f"lr {lr:.2e}"
                )

            if config.save_checkpoints and it > 0 and it % config.ckpt_interval == 0:
                ckpt_path = ckpt_dir / f"ckpt_iter_{it}.pt"

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=it,
                    out=ckpt_path,
                )

                artifact = wandb.Artifact(
                    name=f"{run.id}-ckpt-iter-{it}",
                    type="model",
                )
                artifact.add_file(str(ckpt_path))
                run.log_artifact(artifact)

        if config.save_checkpoints:
            final_ckpt_path = ckpt_dir / "ckpt_final.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=config.max_iters,
                out=final_ckpt_path,
            )


def main():
    train()


if __name__ == "__main__":
    main()
