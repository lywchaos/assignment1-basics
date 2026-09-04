import os
from collections import Counter, defaultdict
from collections.abc import Iterable

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
WordCounts = dict[Word, int]


def prepare_docs(input_path: str | os.PathLike, special_tokens: list[str]) -> list[str]:
    with open(input_path, encoding="utf-8") as file:
        corpus = file.read()

    if not special_tokens:
        return [corpus]
    if any(token == "" for token in special_tokens):
        raise ValueError("special_tokens must not contain empty strings")

    pattern = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return regex.split(pattern, corpus)


def pretokenize(doc: str, pat: str = PAT) -> WordCounts:
    ret: Counter[Word] = Counter()
    for match in regex.finditer(pat, doc):
        token = match.group().encode("utf-8")
        token_seq = tuple(bytes([byte]) for byte in token)
        ret[token_seq] += 1
    return dict(ret)


def merge_counter(counters: Iterable[WordCounts]) -> WordCounts:
    ret: Counter[Word] = Counter()
    for counter in counters:
        ret.update(counter)
    return dict(ret)


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    ret = {i: st.encode() for i, st in enumerate(special_tokens)}
    cur_len = len(ret)
    for i in range(256):
        ret[cur_len + i] = bytes([i])
    return ret


def merge_word(word: Word, pair: Pair) -> Word:
    merged_word: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged_word.append(pair[0] + pair[1])
            i += 2
        else:
            merged_word.append(word[i])
            i += 1
    return tuple(merged_word)


def apply_merge(token_seq_counter: WordCounts, max_pair: Pair) -> WordCounts:
    ret: defaultdict[Word, int] = defaultdict(int)
    for word, count in token_seq_counter.items():
        if max_pair in zip(word, word[1:]):
            word = merge_word(word, max_pair)
        ret[word] += count
    return dict(ret)


def train(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Pair]]:
    vocab = init_vocab(special_tokens)
    len_init_vocab = len(vocab)
    if vocab_size < len_init_vocab:
        raise ValueError(f"vocab_size must be at least {len_init_vocab}")

    docs = prepare_docs(input_path, special_tokens)
    token_seq_counter = merge_counter(pretokenize(doc) for doc in docs)

    merges: list[Pair] = []
    for _ in range(len_init_vocab, vocab_size):
        pair_counter: defaultdict[Pair, int] = defaultdict(int)
        for word, count in token_seq_counter.items():
            for pair in zip(word, word[1:]):
                pair_counter[pair] += count

        if not pair_counter:
            break
        max_pair = max(pair_counter.items(), key=lambda item: (item[1], item[0]))[0]

        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        token_seq_counter = apply_merge(token_seq_counter, max_pair)

    return vocab, merges
