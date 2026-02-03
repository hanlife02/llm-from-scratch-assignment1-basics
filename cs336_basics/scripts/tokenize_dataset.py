from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from cs336_basics.scripts.bpe_utils import resolve_input_path, resolve_name
from cs336_basics.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a dataset into uint16 numpy arrays.")
    parser.add_argument("--dataset", choices=["tinystories", "owt"], default="tinystories")
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--merges-path", default=None)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--chunk-size", type=int, default=1 << 20)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def iter_tokens(path: Path, tokenizer: BPETokenizer, chunk_size: int, *, show_progress: bool):
    total_bytes = path.stat().st_size
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        def chunk_iter():
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        iterable = chunk_iter()
        if show_progress:
            iterable = tqdm(iterable, total=max(1, total_bytes // chunk_size), desc=f"Reading {path.name}")
        yield from tokenizer.encode_iterable(iterable)


def main() -> None:
    args = parse_args()

    input_path = resolve_input_path(args.input_path, args.dataset, args.split)
    name = resolve_name(None, args.dataset if args.input_path is None else None, args.split, input_path)
    if args.dataset and args.input_path is None:
        bpe_name = f"{args.dataset}_train"
    else:
        bpe_name = name
    vocab_path = Path(args.vocab_path) if args.vocab_path else Path("artifacts/bpe") / f"{bpe_name}_vocab.json"
    merges_path = Path(args.merges_path) if args.merges_path else Path("artifacts/bpe") / f"{bpe_name}_merges.json"
    out_path = Path(args.out_path) if args.out_path else Path("data/processed") / f"{name}.npy"

    special_tokens = list(args.special_token) if args.special_token else ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=special_tokens)

    show_progress = not args.no_progress
    token_iter = iter_tokens(input_path, tokenizer, args.chunk_size, show_progress=show_progress)
    total_tokens = sum(1 for _ in token_iter)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(total_tokens,))

    token_iter = iter_tokens(input_path, tokenizer, args.chunk_size, show_progress=show_progress)
    for i, token_id in enumerate(token_iter):
        if token_id > np.iinfo(np.uint16).max:
            raise ValueError("Token id exceeds uint16 range")
        memmap[i] = token_id

    memmap.flush()

    stats = {
        "input_path": str(input_path),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
        "out_path": str(out_path),
        "num_tokens": int(total_tokens),
    }
    stats_path = out_path.with_suffix(".json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    print(f"Saved tokens to {out_path}")
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
