from __future__ import annotations

import json
from pathlib import Path

DATASET_DEFAULTS = {
    "tinystories": {
        "train": "data/TinyStoriesV2-GPT4-train.txt",
        "valid": "data/TinyStoriesV2-GPT4-valid.txt",
        "vocab_size": 10000,
    },
    "owt": {
        "train": "data/owt_train.txt",
        "valid": "data/owt_valid.txt",
        "vocab_size": 32000,
    },
}


def resolve_input_path(input_path: str | None, dataset: str | None, split: str) -> Path:
    if input_path:
        return Path(input_path)
    if dataset is None:
        raise ValueError("dataset or input_path must be provided")
    dataset_cfg = DATASET_DEFAULTS.get(dataset)
    if dataset_cfg is None:
        raise ValueError(f"unknown dataset '{dataset}'")
    if split not in dataset_cfg:
        raise ValueError(f"unknown split '{split}' for dataset '{dataset}'")
    return Path(dataset_cfg[split])


def resolve_vocab_size(vocab_size: int | None, dataset: str | None) -> int:
    if vocab_size is not None:
        return vocab_size
    if dataset is None:
        raise ValueError("vocab_size must be provided when dataset is not set")
    dataset_cfg = DATASET_DEFAULTS.get(dataset)
    if dataset_cfg is None:
        raise ValueError(f"unknown dataset '{dataset}'")
    return int(dataset_cfg["vocab_size"])


def resolve_name(name: str | None, dataset: str | None, split: str, input_path: Path) -> str:
    if name:
        return name
    if dataset:
        return f"{dataset}_{split}"
    return input_path.stem


def encode_token(token: bytes) -> str:
    return token.decode("latin-1")


def decode_token(token: str) -> bytes:
    return token.encode("latin-1")


def write_vocab(path: Path, vocab: dict[int, bytes]) -> None:
    ordered = [encode_token(vocab[i]) for i in range(len(vocab))]
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=True, indent=2)


def write_merges(path: Path, merges: list[tuple[bytes, bytes]]) -> None:
    encoded = [[encode_token(a), encode_token(b)] for a, b in merges]
    with path.open("w", encoding="utf-8") as f:
        json.dump(encoded, f, ensure_ascii=True, indent=2)


def read_vocab(path: Path) -> list[bytes]:
    with path.open("r", encoding="utf-8") as f:
        tokens = json.load(f)
    return [decode_token(token) for token in tokens]
