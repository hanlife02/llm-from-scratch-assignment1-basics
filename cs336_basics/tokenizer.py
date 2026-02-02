from __future__ import annotations

import regex as re

PRETOKENIZER = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self._bpe_cache: dict[bytes, list[bytes]] = {}

        if self.special_tokens:
            existing = set(self.vocab.values())
            next_id = max(self.vocab.keys(), default=-1) + 1
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in existing:
                    self.vocab[next_id] = token_bytes
                    existing.add(token_bytes)
                    next_id += 1

        self.id_to_token = self.vocab
        self.token_to_id = {v: k for k, v in self.vocab.items()}

        if self.special_tokens:
            specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(s) for s in specials_sorted) + ")"
            self._special_re = re.compile(pattern)
        else:
            self._special_re = None
        self._special_prefixes: set[str] = set()
        self._special_prefix_max_len = 0
        if self.special_tokens:
            for special in self.special_tokens:
                for i in range(1, len(special)):
                    prefix = special[:i]
                    self._special_prefixes.add(prefix)
                    if len(prefix) > self._special_prefix_max_len:
                        self._special_prefix_max_len = len(prefix)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        import json
        from pathlib import Path

        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)

        with vocab_path.open("r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        if isinstance(vocab_data, list):
            vocab = {i: token.encode("latin-1") for i, token in enumerate(vocab_data)}
        elif isinstance(vocab_data, dict):
            if all(isinstance(k, str) for k in vocab_data.keys()):
                vocab = {int(v): k.encode("latin-1") for k, v in vocab_data.items()}
            else:
                vocab = {int(k): v.encode("latin-1") for k, v in vocab_data.items()}
        else:
            raise ValueError("Unsupported vocab format")

        if merges_path.suffix == ".json":
            with merges_path.open("r", encoding="utf-8") as f:
                merges_data = json.load(f)
            merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in merges_data]
        else:
            merges = []
            with merges_path.open("r", encoding="utf-8") as f:
                for line in f:
                    cleaned = line.rstrip()
                    if not cleaned:
                        continue
                    parts = cleaned.split(" ")
                    if len(parts) != 2:
                        continue
                    merges.append((parts[0].encode("latin-1"), parts[1].encode("latin-1")))

        return cls(vocab, merges, special_tokens)

    def _split_special(self, text: str) -> list[str]:
        if not self._special_re:
            return [text]
        parts = self._special_re.split(text)
        return [p for p in parts if p != ""]

    def _special_holdback_len(self, text: str) -> int:
        if not self._special_prefixes:
            return 0
        max_len = min(self._special_prefix_max_len, len(text))
        for length in range(max_len, 0, -1):
            if text[-length:] in self._special_prefixes:
                return length
        return 0

    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        return set(zip(tokens, tokens[1:]))

    def _merge_pair(self, tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        merged = []
        i = 0
        a, b = pair
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        cached = self._bpe_cache.get(token_bytes)
        if cached is not None:
            return cached

        tokens = [bytes([b]) for b in token_bytes]
        pairs = self._get_pairs(tokens)
        while pairs:
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            tokens = self._merge_pair(tokens, best)
            pairs = self._get_pairs(tokens)

        self._bpe_cache[token_bytes] = tokens
        return tokens

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for segment in self._split_special(text):
            if segment in self.special_tokens:
                ids.append(self.token_to_id[segment.encode("utf-8")])
                continue
            for token in PRETOKENIZER.findall(segment):
                for bpe_tok in self._bpe(token.encode("utf-8")):
                    ids.append(self.token_to_id[bpe_tok])
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.id_to_token[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        carry = ""
        for chunk in iterable:
            carry += chunk
            holdback = self._special_holdback_len(carry)
            if holdback:
                text, carry = carry[:-holdback], carry[-holdback:]
            else:
                text, carry = carry, ""

            for segment in self._split_special(text):
                if segment in self.special_tokens:
                    yield self.token_to_id[segment.encode("utf-8")]
                    continue
                for token in PRETOKENIZER.findall(segment):
                    for bpe_tok in self._bpe(token.encode("utf-8")):
                        yield self.token_to_id[bpe_tok]

        if carry:
            for segment in self._split_special(carry):
                if segment in self.special_tokens:
                    yield self.token_to_id[segment.encode("utf-8")]
                    continue
                for token in PRETOKENIZER.findall(segment):
                    for bpe_tok in self._bpe(token.encode("utf-8")):
                        yield self.token_to_id[bpe_tok]
