from __future__ import annotations

import argparse
import json
import resource
import sys
import time
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
    parser.add_argument("--force-progress", action="store_true")
    parser.add_argument("--stats-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = resolve_input_path(args.input_path, args.dataset, args.split)
    vocab_size = resolve_vocab_size(args.vocab_size, args.dataset if args.input_path is None else None)
    special_tokens = list(args.special_token) if args.special_token else ["<|endoftext|>"]

    start_time = time.perf_counter()
    vocab, merges = train_bpe(
        str(input_path),
        vocab_size,
        special_tokens,
        progress=not args.no_progress,
        force_progress=args.force_progress,
    )
    elapsed_seconds = time.perf_counter() - start_time
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = resolve_name(args.name, args.dataset if args.input_path is None else None, args.split, input_path)
    vocab_path = out_dir / f"{name}_vocab.json"
    merges_path = out_dir / f"{name}_merges.json"

    write_vocab(vocab_path, vocab)
    write_merges(merges_path, merges)

    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")

    max_id = max(range(len(vocab)), key=lambda i: len(vocab[i]))
    max_token = vocab[max_id]
    preview = max_token.decode("utf-8", errors="replace")

    stats = {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hours": elapsed_seconds / 3600.0,
        "peak_rss_kb": max_rss_kb,
        "peak_rss_gb": max_rss_kb / (1024.0 * 1024.0),
        "longest_token_id": max_id,
        "longest_token_length_bytes": len(max_token),
        "longest_token_preview_utf8": preview,
        "longest_token_bytes_repr": repr(max_token),
    }

    stats_path = Path(args.stats_out) if args.stats_out else out_dir / f"{name}_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    print(f"Elapsed seconds: {elapsed_seconds:.2f} ({elapsed_seconds / 3600.0:.4f} hours)")
    print(f"Peak RSS: {max_rss_kb / (1024.0 * 1024.0):.3f} GB")
    print(f"Longest token id: {max_id}")
    print(f"Longest token length (bytes): {len(max_token)}")
    print(f"Longest token preview (utf-8): {preview}")
    print(f"Longest token bytes repr: {max_token!r}")
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
