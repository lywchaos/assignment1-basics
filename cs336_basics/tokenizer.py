import os
from collections import defaultdict

import regex as re


def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    special_tokens = special_tokens or []
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i, special_tokens in enumerate(special_tokens):
        vocab[i + 256] = special_tokens.encode()
    return vocab


def pre_tokenize(text: str) -> list[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(PAT, text)


def test_pre_tokenize():
    text = "some text that i'll pre-tokenize"
    assert pre_tokenize(text) == [
        "some",
        " text",
        " that",
        " i",
        "'ll",
        " pre",
        "-",
        "tokenize",
    ]


def split_by_special_token(text: str, special_tokens: list[str]) -> list[str]:
    pat = r"|".join([re.escape(t) for t in special_tokens])
    return re.split(pat, text)


def test_split_by_special_token():
    text = "a<|a|>b<|c|>d"
    special_tokens = ["<|a|>", "<|c|>"]
    assert split_by_special_token(text, special_tokens) == ["a", "b", "d"]


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    vocab = init_vocab(special_tokens)
    assert vocab_size > len(vocab)
    nr_merges = vocab_size - len(vocab)
    merges = []

    with open(input_path) as f:
        text = f.read()
    tokens = []
    for chunk in split_by_special_token(text, special_tokens):
        tokens.extend(pre_tokenize(chunk))
    token_bytes_counter = defaultdict(int)
    for t in tokens:
        token_bytes_counter[tuple([bytes([i]) for i in t.encode()])] += 1

    for _ in range(nr_merges):
        byte_pair_counter = defaultdict(int)
        for token_bytes, count in token_bytes_counter.items():
            for byte1, byte2 in zip(token_bytes, token_bytes[1:]):
                byte_pair_counter[(byte1, byte2)] += count
        max_value = max(byte_pair_counter.values())
        best_pairs = [k for k, v in byte_pair_counter.items() if v == max_value]
        best_pair = max(best_pairs)
        merges.append(best_pair)
        vocab[len(vocab)] = b"".join(best_pair)

        new_token_bytes_counter = defaultdict(int)
        for token_bytes, count in token_bytes_counter.items():
            i = 0
            new_token = []
            while i < len(token_bytes):
                if (
                    i < len(token_bytes) - 1
                    and (token_bytes[i], token_bytes[i + 1]) == best_pair
                ):
                    new_token.append(b"".join(best_pair))
                    i += 2
                else:
                    new_token.append(token_bytes[i])
                    i += 1
            new_token_tuple = tuple(new_token)
            new_token_bytes_counter[new_token_tuple] += count
        token_bytes_counter = new_token_bytes_counter

    return vocab, merges
