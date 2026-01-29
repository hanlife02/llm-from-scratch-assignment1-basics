from __future__ import annotations

from collections import Counter, defaultdict

import regex as re

from .tokenizer import PRETOKENIZER


def _build_special_splitter(special_tokens: list[str]):
    if not special_tokens:
        return None, None
    specials_sorted = sorted(special_tokens, key=len, reverse=True)
    pattern = "(" + "|".join(re.escape(s) for s in specials_sorted) + ")"
    special_re = re.compile(pattern)
    special_set = set(special_tokens)
    return special_re, special_set


def _split_special(text: str, special_re: re.Pattern | None) -> list[str]:
    if not special_re:
        return [text]
    parts = special_re.split(text)
    return [p for p in parts if p != ""]


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.
    Returns vocab mapping (id -> bytes) and list of merges (bytes, bytes).
    """
    special_tokens = special_tokens or []
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must include base bytes and special tokens")

    special_re, special_set = _build_special_splitter(special_tokens)
    byte_symbols = [bytes([i]) for i in range(256)]
    word_freq: Counter[tuple[bytes, ...]] = Counter()

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    for segment in _split_special(text, special_re):
        if special_set and segment in special_set:
            word_freq[(segment.encode("utf-8"),)] += 1
            continue
        for token in PRETOKENIZER.findall(segment):
            token_bytes = token.encode("utf-8")
            word = tuple(byte_symbols[b] for b in token_bytes)
            word_freq[word] += 1

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - (256 + len(special_tokens))

    words = [list(word) for word in word_freq.keys()]
    freqs = list(word_freq.values())
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for idx, (word, freq) in enumerate(zip(words, freqs)):
        if len(word) < 2:
            continue
        for pair in zip(word, word[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            pair_to_words[pair].add(idx)

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Tie-break by lexicographic order of the pair for determinism.
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(best_pair)
        a, b = best_pair

        affected = list(pair_to_words.pop(best_pair, set()))
        for idx in affected:
            word = words[idx]
            freq = freqs[idx]
            if len(word) < 2:
                continue

            old_pairs = list(zip(word, word[1:]))
            for pair in old_pairs:
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            for pair in set(old_pairs):
                pair_words = pair_to_words.get(pair)
                if pair_words is not None:
                    pair_words.discard(idx)
                    if not pair_words:
                        del pair_to_words[pair]

            merged: list[bytes] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[idx] = merged

            if len(merged) < 2:
                continue
            new_pairs = list(zip(merged, merged[1:]))
            for pair in new_pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
            for pair in set(new_pairs):
                pair_to_words[pair].add(idx)

    vocab_list: list[bytes] = []
    for token in special_tokens:
        vocab_list.append(token.encode("utf-8"))
    vocab_list.extend(byte_symbols)
    for a, b in merges:
        vocab_list.append(a + b)

    vocab = {i: tok for i, tok in enumerate(vocab_list)}
    return vocab, merges
