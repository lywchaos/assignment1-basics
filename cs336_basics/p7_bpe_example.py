from collections import Counter

from rich import print

CORPUS = """low low low low low lower lower widest widest widest newest newest newest newest newest newest"""

VOCAB = [
    "<|endoftext|>",
    *[bytes([i]) for i in range(256)],
]


def pretokenize(corpus: str) -> dict[tuple[bytes, ...], int]:
    # 先按 str 计数，再把去重后的 word 转成 token 元组（str 哈希更快，等价 word 只转一次）
    counter = Counter(corpus.split(" "))
    return {tuple(bytes([ord(ch)]) for ch in w): c for w, c in counter.items()}


def _apply_merge_in_token(pair: tuple[bytes, ...], token: tuple[bytes, ...]) -> tuple[bytes, ...]:
    new_token = []
    i = 0
    while i < len(token):
        if i + 1 < len(token) and (token[i], token[i + 1]) == pair:
            new_token.append(token[i] + token[i + 1])
            i += 2
        else:
            new_token.append(token[i])
            i += 1
    return tuple(new_token)


def _find_max_pair_in_counter(counter: Counter) -> tuple[bytes, bytes]:
    max_count = max(counter.most_common(), key=lambda c: c[1])[1]
    pairs = [k for k in counter.keys() if counter[k] == max_count]
    return max(pairs)


def merge(vocab: list, vocab_size: int, corpus: str) -> None:
    token_counter = Counter(pretokenize(corpus))

    original_len_vocab = len(vocab)
    for r in range(original_len_vocab, vocab_size):
        print(f"{token_counter=}")
        pair_counter = Counter()
        for token in token_counter:
            pairs = zip(token, token[1:])
            pair_counter.update({pair: token_counter[token] for pair in pairs})

        max_pair = _find_max_pair_in_counter(pair_counter)
        vocab.append(max_pair[0] + max_pair[1])
        print(f"{max_pair=}")

        new_token_counter = Counter()
        for t, c in token_counter.items():
            _pairs = tuple(zip(t, t[1:]))
            if max_pair in _pairs:
                new_t = _apply_merge_in_token(max_pair, t)
                new_token_counter.update({new_t: c})
            else:
                new_token_counter.update({t: c})

        token_counter = new_token_counter


if __name__ == "__main__":
    merge(VOCAB, 264, CORPUS)
