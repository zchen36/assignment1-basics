from torch import Tensor, nn
from datetime import datetime
from pathlib import Path
from jaxtyping import Float, Int
import torch
from llm.softmax import softmax
from tokenizer.tokenizer import Tokenizer
from llm.attention import Transformer
from llm.adamW import AdamW
from llm.checkpoint import load_checkpoint

from tokenizer.constants import (
    TS_VOCAB_PATH,
    TS_MERGES_PATH,
)


def top_p_sample(
    logits: Float[Tensor, "vocab_size"],
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Int[Tensor, ""]:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)  # return index, which is the token id

    logits = logits / temperature

    sorted_logits, sorted_indics = torch.sort(logits, descending=True)
    probs = softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(probs, dim=-1)

    cutoff = cumulative_probs > top_p
    cutoff[0] = False

    sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))

    filtered_probs = softmax(sorted_logits, dim=-1)
    sampled_sorted_index = torch.multinomial(filtered_probs, num_samples=1)

    token_id = sorted_indics[sampled_sorted_index]

    return token_id.squeeze(0)


def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    endoftext_token: str = "<|endoftext|>",
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "mps",
) -> str:
    model.eval()

    endoftext_id = tokenizer.encode(endoftext_token)[0]

    token_ids = tokenizer.encode(prompt)
    x = torch.tensor(token_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[-context_length:].unsqueeze(0)  # [batch, seq_length]

            logits = model(x_cond)  # [batch, seq_len, vocab_size]

            next_logits = logits[0, -1, :]  # [vocab_size]

            next_token = top_p_sample(
                logits=next_logits,
                temperature=temperature,
                top_p=top_p,
            )

            x = torch.cat([x, next_token.view(1)], dim=0)

            if next_token.item() == endoftext_id:
                break

    return tokenizer.decode(x.tolist())


if __name__ == "__main__":
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

    device = "mps"
    ckpt_path = ROOT / "data/ckpt/2026_05_03_18_17/ckpt_final.pt"

    ts_tokenizer = Tokenizer.from_file(
        vocab_path=TS_VOCAB_PATH,
        merges_path=TS_MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )

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

    optimizer = AdamW(
        params=model.parameters(),
        lr=config["max_lr"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
        eps=config["optimizer_eps"],
    )

    load_checkpoint(src=ckpt_path, model=model, optimizer=optimizer)

    model.to(device)

    prompt = "Once upon a time"

    text = generate(
        model=model,
        tokenizer=ts_tokenizer,
        prompt=prompt,
        max_new_tokens=200,
        context_length=config["context_length"],
        temperature=1.0,
        top_p=0.9,
        device=device,
    )

    print(text)
