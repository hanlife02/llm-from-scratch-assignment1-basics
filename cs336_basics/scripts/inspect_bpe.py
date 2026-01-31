from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from cs336_basics.scripts.bpe_utils import read_vocab, resolve_input_path, resolve_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a serialized BPE vocab.")
    parser.add_argument("--dataset", choices=["tinystories", "owt"], default="tinystories")
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--out-dir", default="artifacts/bpe")
    parser.add_argument("--name", default=None)
    parser.add_argument("--vocab", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.vocab:
        vocab_path = Path(args.vocab)
    else:
        input_path = resolve_input_path(None, args.dataset, args.split)
        name = resolve_name(args.name, args.dataset, args.split, input_path)
        vocab_path = Path(args.out_dir) / f"{name}_vocab.json"

    vocab = read_vocab(vocab_path)
    max_id = max(range(len(vocab)), key=lambda i: len(vocab[i]))
    max_token = vocab[max_id]

    preview = max_token.decode("utf-8", errors="replace")
    print(f"Longest token id: {max_id}")
    print(f"Length (bytes): {len(max_token)}")
    print(f"Token preview (utf-8): {preview}")
    print(f"Token bytes repr: {max_token!r}")


if __name__ == "__main__":
    main()
