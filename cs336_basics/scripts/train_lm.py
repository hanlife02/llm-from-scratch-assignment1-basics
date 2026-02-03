from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from cs336_basics.data import get_batch
from cs336_basics.decoding import generate
from cs336_basics.experiment_log import ExperimentLogger
from cs336_basics.optim import get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.transformer import TransformerConfig, TransformerLM
from cs336_basics.nn_utils import gradient_clipping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument("--dataset", choices=["tinystories", "owt"], default="tinystories")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--valid-path", default=None)
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--merges-path", default=None)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--post-norm", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--cosine-iters", type=int, default=20000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-dir", default="artifacts/logs")
    parser.add_argument("--save-dir", default="artifacts/checkpoints")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--device", default=None)
    parser.add_argument("--use-wandb", action="store_true")

    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.train_path:
        train_path = Path(args.train_path)
    else:
        train_path = Path("data/processed") / f"{args.dataset}_train.npy"

    if args.valid_path:
        valid_path = Path(args.valid_path)
    else:
        valid_path = Path("data/processed") / f"{args.dataset}_valid.npy"

    bpe_name = f"{args.dataset}_train"
    vocab_path = Path(args.vocab_path) if args.vocab_path else Path("artifacts/bpe") / f"{bpe_name}_vocab.json"
    merges_path = Path(args.merges_path) if args.merges_path else Path("artifacts/bpe") / f"{bpe_name}_merges.json"
    return train_path, valid_path, vocab_path, merges_path


def load_dataset(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return np.load(path, mmap_mode="r")


def vocab_size_from_file(vocab_path: Path) -> int:
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    if isinstance(vocab_data, list):
        return len(vocab_data)
    if isinstance(vocab_data, dict):
        return len(vocab_data)
    raise ValueError("Unsupported vocab format")


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    data: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    vocab_size: int,
    device: str,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / len(losses))


def main() -> None:
    args = parse_args()
    train_path, valid_path, vocab_path, merges_path = resolve_paths(args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_data = load_dataset(train_path)
    valid_data = load_dataset(valid_path)
    vocab_size = vocab_size_from_file(vocab_path)

    config = TransformerConfig(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        pre_norm=not args.post_norm,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
        dropout=args.dropout,
    )
    model = TransformerLM(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer) + 1

    special_tokens = list(args.special_token) if args.special_token else ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=special_tokens)

    if args.generate:
        if args.prompt:
            prompt_ids = tokenizer.encode(args.prompt)
        else:
            prompt_ids = [tokenizer.token_to_id[special_tokens[0].encode("utf-8")]]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        tokens = generate(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            context_length=args.context_length,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.token_to_id.get(special_tokens[0].encode("utf-8")),
        )
        print(tokenizer.decode(tokens[0].tolist()))
        return

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(
        log_dir / f"{args.run_name}.csv",
        use_wandb=args.use_wandb,
        run_name=args.run_name,
        config=vars(args),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step_time = 0.0
    for step in range(start_step, args.max_steps):
        t0 = time.perf_counter()
        lr = get_lr_cosine_schedule(
            step,
            args.lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        step_time = time.perf_counter() - t0

        if step % args.log_interval == 0:
            logger.log(
                step,
                train_loss=float(loss.item()),
                val_loss=float("nan"),
                lr=float(lr),
                step_time=step_time,
            )

        if step % args.eval_interval == 0:
            val_loss = estimate_loss(
                model,
                valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                eval_iters=args.eval_iters,
                vocab_size=vocab_size,
                device=device,
            )
            logger.log(
                step,
                train_loss=float(loss.item()),
                val_loss=val_loss,
                lr=float(lr),
                step_time=step_time,
            )

        if step != 0 and step % args.save_interval == 0:
            ckpt_path = save_dir / f"{args.run_name}_step{step}.pt"
            save_checkpoint(model, optimizer, step, ckpt_path)

    final_ckpt = save_dir / f"{args.run_name}_final.pt"
    save_checkpoint(model, optimizer, args.max_steps - 1, final_ckpt)
    logger.close()


if __name__ == "__main__":
    main()
