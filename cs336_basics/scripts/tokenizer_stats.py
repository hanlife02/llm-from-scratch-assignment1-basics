from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from cs336_basics.scripts.bpe_utils import DATASET_DEFAULTS, resolve_input_path
from cs336_basics.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute tokenizer compression ratios and throughput.")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--special-token", default="<|endoftext|>")
    parser.add_argument("--throughput-bytes", type=int, default=10_000_000)
    parser.add_argument("--out-dir", default="artifacts/bpe")
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--tinystories-vocab", default=None)
    parser.add_argument("--tinystories-merges", default=None)
    parser.add_argument("--owt-vocab", default=None)
    parser.add_argument("--owt-merges", default=None)
    return parser.parse_args()


def iter_documents(path: Path, special_token: str):
    buffer = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            buffer += line
            while True:
                idx = buffer.find(special_token)
                if idx == -1:
                    break
                yield buffer[:idx]
                buffer = buffer[idx + len(special_token) :]
        if buffer:
            yield buffer


def reservoir_sample(docs, k: int, seed: int) -> list[str]:
    random.seed(seed)
    sample: list[str] = []
    for i, doc in enumerate(docs):
        if i < k:
            sample.append(doc)
        else:
            j = random.randint(0, i)
            if j < k:
                sample[j] = doc
    return sample


def compression_ratio(tokenizer: BPETokenizer, docs: list[str]) -> tuple[float, int, int]:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    return total_bytes / total_tokens, total_bytes, total_tokens


def measure_throughput(tokenizer: BPETokenizer, input_path: Path, max_bytes: int) -> tuple[float, int, float]:
    buffer = ""
    collected_bytes = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_bytes = len(line.encode("utf-8"))
            if collected_bytes + line_bytes > max_bytes:
                remaining = max_bytes - collected_bytes
                if remaining > 0:
                    buffer += line.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
                    collected_bytes = max_bytes
                break
            buffer += line
            collected_bytes += line_bytes
            if collected_bytes >= max_bytes:
                break

    start = time.perf_counter()
    _ = tokenizer.encode(buffer)
    elapsed = time.perf_counter() - start
    throughput = collected_bytes / elapsed if elapsed > 0 else 0.0
    return throughput, collected_bytes, elapsed


def default_tokenizer_paths(out_dir: Path, dataset: str) -> tuple[Path, Path]:
    name = f"{dataset}_train"
    vocab_path = out_dir / f"{name}_vocab.json"
    merges_path = out_dir / f"{name}_merges.json"
    return vocab_path, merges_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    tinystories_vocab = (
        Path(args.tinystories_vocab)
        if args.tinystories_vocab
        else default_tokenizer_paths(out_dir, "tinystories")[0]
    )
    tinystories_merges = (
        Path(args.tinystories_merges)
        if args.tinystories_merges
        else default_tokenizer_paths(out_dir, "tinystories")[1]
    )
    owt_vocab = Path(args.owt_vocab) if args.owt_vocab else default_tokenizer_paths(out_dir, "owt")[0]
    owt_merges = Path(args.owt_merges) if args.owt_merges else default_tokenizer_paths(out_dir, "owt")[1]

    special_tokens = [args.special_token]
    tiny_tokenizer = BPETokenizer.from_files(str(tinystories_vocab), str(tinystories_merges), special_tokens)
    owt_tokenizer = BPETokenizer.from_files(str(owt_vocab), str(owt_merges), special_tokens)

    tinystories_path = resolve_input_path(None, "tinystories", "train")
    owt_path = resolve_input_path(None, "owt", "train")

    tiny_docs = reservoir_sample(iter_documents(tinystories_path, args.special_token), args.samples, args.seed)
    owt_docs = reservoir_sample(iter_documents(owt_path, args.special_token), args.samples, args.seed)

    tiny_ratio, tiny_bytes, tiny_tokens = compression_ratio(tiny_tokenizer, tiny_docs)
    owt_ratio, owt_bytes, owt_tokens = compression_ratio(owt_tokenizer, owt_docs)
    owt_on_tiny_ratio, owt_on_tiny_bytes, owt_on_tiny_tokens = compression_ratio(tiny_tokenizer, owt_docs)

    tiny_throughput, tiny_tp_bytes, tiny_tp_seconds = measure_throughput(
        tiny_tokenizer, tinystories_path, args.throughput_bytes
    )
    owt_throughput, owt_tp_bytes, owt_tp_seconds = measure_throughput(
        owt_tokenizer, owt_path, args.throughput_bytes
    )

    pile_bytes = 825 * 1_000_000_000
    pile_hours_tiny = pile_bytes / tiny_throughput / 3600 if tiny_throughput else None
    pile_hours_owt = pile_bytes / owt_throughput / 3600 if owt_throughput else None

    stats = {
        "samples": args.samples,
        "special_token": args.special_token,
        "compression_ratio_bytes_per_token": {
            "tinystories_tokenizer_on_tinystories": tiny_ratio,
            "owt_tokenizer_on_owt": owt_ratio,
            "tinystories_tokenizer_on_owt": owt_on_tiny_ratio,
        },
        "compression_raw": {
            "tinystories": {"bytes": tiny_bytes, "tokens": tiny_tokens},
            "owt": {"bytes": owt_bytes, "tokens": owt_tokens},
            "owt_on_tinystories_tokenizer": {"bytes": owt_on_tiny_bytes, "tokens": owt_on_tiny_tokens},
        },
        "throughput_bytes_per_second": {
            "tinystories_tokenizer_on_tinystories": tiny_throughput,
            "owt_tokenizer_on_owt": owt_throughput,
        },
        "throughput_samples": {
            "tinystories": {"bytes": tiny_tp_bytes, "seconds": tiny_tp_seconds},
            "owt": {"bytes": owt_tp_bytes, "seconds": owt_tp_seconds},
        },
        "pile_estimate_hours": {
            "tinystories_tokenizer": pile_hours_tiny,
            "owt_tokenizer": pile_hours_owt,
        },
    }

    stats_path = Path(args.stats_out) if args.stats_out else out_dir / "tokenizer_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    print(f"TinyStories tokenizer compression (bytes/token): {tiny_ratio:.4f}")
    print(f"OWT tokenizer compression (bytes/token): {owt_ratio:.4f}")
    print(f"TinyStories tokenizer on OWT compression (bytes/token): {owt_on_tiny_ratio:.4f}")
    print(f"TinyStories tokenizer throughput (bytes/sec): {tiny_throughput:.2f}")
    print(f"OWT tokenizer throughput (bytes/sec): {owt_throughput:.2f}")
    if pile_hours_tiny is not None:
        print(f"Pile (825GB) estimate with TinyStories tokenizer: {pile_hours_tiny:.2f} hours")
    if pile_hours_owt is not None:
        print(f"Pile (825GB) estimate with OWT tokenizer: {pile_hours_owt:.2f} hours")
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
