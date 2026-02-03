from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from tqdm.auto import tqdm

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

    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--post-norm", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=4096)
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
    parser.add_argument("--no-progress", action="store_true")

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


class _NullLogger:
    def log(self, *args: object, **kwargs: object) -> None:
        return None

    def close(self) -> None:
        return None


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def maybe_launch_ddp(args: argparse.Namespace) -> bool:
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        return False
    if args.generate:
        return False
    if args.device and not args.device.startswith("cuda"):
        return False
    if args.device and args.device.startswith("cuda:"):
        return False
    if not torch.cuda.is_available():
        return False
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        return False

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(find_free_port()))
    os.environ["WORLD_SIZE"] = str(world_size)
    mp.spawn(_distributed_worker, nprocs=world_size, args=(args, world_size))
    return True


def _distributed_worker(local_rank: int, args: argparse.Namespace, world_size: int) -> None:
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    run_training(args)


def init_distributed(device_override: str | None) -> tuple[bool, int, int, int, str]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device_override:
            if device_override.startswith("cuda"):
                device = f"cuda:{local_rank}"
            else:
                device = device_override
        else:
            device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        if device.startswith("cuda"):
            torch.cuda.set_device(device)

        if world_size > 1:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            return True, rank, local_rank, world_size, device
        return False, rank, local_rank, world_size, device

    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 0, 1, device


def autotune_batch_size(
    model: torch.nn.Module,
    data: np.ndarray,
    *,
    context_length: int,
    vocab_size: int,
    device: str,
    max_batch_size: int,
) -> int:
    if not device.startswith("cuda"):
        return 1

    def try_batch_size(batch_size: int) -> bool:
        try:
            model.zero_grad(set_to_none=True)
            x, y = get_batch(data, batch_size, context_length, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            del x, y, logits, loss
            torch.cuda.synchronize()
            return True
        except RuntimeError as err:
            if "out of memory" in str(err).lower():
                torch.cuda.empty_cache()
                return False
            raise

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    low = 1
    if not try_batch_size(low):
        raise RuntimeError("Batch size 1 does not fit on this GPU.")

    high = low
    while high < max_batch_size:
        candidate = min(high * 2, max_batch_size)
        if try_batch_size(candidate):
            low = candidate
            high = candidate
            if high == max_batch_size:
                return high
        else:
            high = candidate
            break

    while low + 1 < high:
        mid = (low + high) // 2
        if try_batch_size(mid):
            low = mid
        else:
            high = mid

    return low


def run_training(args: argparse.Namespace) -> None:
    train_path, valid_path, vocab_path, merges_path = resolve_paths(args)

    distributed, rank, local_rank, world_size, device = init_distributed(args.device)
    is_rank0 = rank == 0
    if not is_rank0:
        args.no_progress = True
        args.use_wandb = False

    torch.manual_seed(args.seed + rank if distributed else args.seed)

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
    if distributed:
        if device.startswith("cuda"):
            model = DDP(model, device_ids=[local_rank])
        else:
            model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model.module if isinstance(model, DDP) else model, optimizer) + 1

    if args.batch_size <= 0 and not args.generate:
        if distributed and world_size > 1 and not is_rank0:
            tuned_batch_size = 1
        else:
            tuned_batch_size = autotune_batch_size(
                model,
                train_data,
                context_length=args.context_length,
                vocab_size=vocab_size,
                device=device,
                max_batch_size=args.max_batch_size,
            )
        if distributed and world_size > 1:
            batch_tensor = torch.tensor([tuned_batch_size], device=device)
            dist.broadcast(batch_tensor, src=0)
            args.batch_size = int(batch_tensor.item())
        else:
            args.batch_size = tuned_batch_size
        if is_rank0:
            print(f"[auto] batch_size={args.batch_size}")

    special_tokens = list(args.special_token) if args.special_token else ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=special_tokens)

    if args.generate:
        if distributed and world_size > 1 and not is_rank0:
            dist.barrier()
            dist.destroy_process_group()
            return
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
        if distributed and world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    if is_rank0:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger: ExperimentLogger | _NullLogger = ExperimentLogger(
            log_dir / f"{args.run_name}.csv",
            use_wandb=args.use_wandb,
            run_name=args.run_name,
            config=vars(args),
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger = _NullLogger()
        save_dir = Path(args.save_dir)

    step_time = 0.0
    last_val_loss: float | None = None
    progress = tqdm(
        total=args.max_steps,
        initial=start_step,
        disable=args.no_progress,
        desc="train",
        unit="step",
    )
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
            if not args.no_progress:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    val_loss="-" if last_val_loss is None else f"{last_val_loss:.4f}",
                    lr=f"{lr:.2e}",
                    step_time=f"{step_time:.2f}s",
                )

        if step % args.eval_interval == 0:
            val_loss = float("nan")
            if not distributed or is_rank0:
                val_loss = estimate_loss(
                    model,
                    valid_data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    eval_iters=args.eval_iters,
                    vocab_size=vocab_size,
                    device=device,
                )
            if distributed and world_size > 1:
                dist.barrier()
            if is_rank0:
                logger.log(
                    step,
                    train_loss=float(loss.item()),
                    val_loss=val_loss,
                    lr=float(lr),
                    step_time=step_time,
                )
                last_val_loss = val_loss
                if not args.no_progress:
                    progress.set_postfix(
                        loss=f"{loss.item():.4f}",
                        val_loss=f"{val_loss:.4f}",
                        lr=f"{lr:.2e}",
                        step_time=f"{step_time:.2f}s",
                    )

        if step != 0 and step % args.save_interval == 0:
            if is_rank0:
                ckpt_path = save_dir / f"{args.run_name}_step{step}.pt"
                save_checkpoint(
                    model.module if isinstance(model, DDP) else model,
                    optimizer,
                    step,
                    ckpt_path,
                )
        progress.update(1)

    if is_rank0:
        final_ckpt = save_dir / f"{args.run_name}_final.pt"
        save_checkpoint(
            model.module if isinstance(model, DDP) else model,
            optimizer,
            args.max_steps - 1,
            final_ckpt,
        )
        logger.close()
    progress.close()
    if distributed and world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if maybe_launch_ddp(args):
        return
    run_training(args)


if __name__ == "__main__":
    main()
