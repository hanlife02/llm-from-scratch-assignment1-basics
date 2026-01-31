from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from cs336_basics.bpe import train_bpe
from cs336_basics.scripts.bpe_utils import resolve_input_path, resolve_name, resolve_vocab_size, write_merges, write_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--dataset", choices=["tinystories", "owt"], default="tinystories")
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--out-dir", default="artifacts/bpe")
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = resolve_input_path(args.input_path, args.dataset, args.split)
    vocab_size = resolve_vocab_size(args.vocab_size, args.dataset if args.input_path is None else None)
    special_tokens = list(args.special_token) if args.special_token else ["<|endoftext|>"]

    vocab, merges = train_bpe(str(input_path), vocab_size, special_tokens, progress=not args.no_progress)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = resolve_name(args.name, args.dataset if args.input_path is None else None, args.split, input_path)
    vocab_path = out_dir / f"{name}_vocab.json"
    merges_path = out_dir / f"{name}_merges.json"

    write_vocab(vocab_path, vocab)
    write_merges(merges_path, merges)

    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")


if __name__ == "__main__":
    main()
