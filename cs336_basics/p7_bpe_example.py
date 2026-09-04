from collections import Counter

from rich import print

CORPUS = """low low low low low lower lower widest widest widest newest newest newest newest newest newest"""

VOCAB = [
    "<|endoftext|>",
    *[bytes([i]) for i in range(256)],
]


def pretokenize(corpus: str) -> dict[tuple[bytes, ...], int]:
    # 先按 str 计数，再把去重后的 word 转成 token 元组（str 哈希更快，等价 word 只转一次）
    # NOTE: bytes([ord(ch)]) 仅因本语料全为 ASCII 才成立，真语料必须走 w.encode("utf-8") 逐字节拆
    counter = Counter(corpus.split(" "))
    return {tuple(bytes([ord(ch)]) for ch in w): c for w, c in counter.items()}


def _apply_merge_in_token(pair: tuple[bytes, bytes], token: tuple[bytes, ...]) -> tuple[bytes, ...]:
    new_token: list[bytes] = []
    i = 0
    while i < len(token):
        if i + 1 < len(token) and (token[i], token[i + 1]) == pair:
            new_token.append(token[i] + token[i + 1])
            i += 2
        else:
            new_token.append(token[i])
            i += 1
    return tuple(new_token)


def _find_max_pair_in_counter(counter: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    # 先筛出全部并列最高频的 pair，再按字典序取大。
    # 两步顺序不可颠倒：先 most_common(1) 截断再取 max，候选集只剩 1 个，tie-break 等于没做。
    max_count = max(counter.values())
    return max(p for p, c in counter.items() if c == max_count)


# NOTE: 真正的 train_bpe 应返回 (vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]])；
# 这里为贴合 handout 的演示，直接原地追加进 vocab，merges 靠 print 观察。
def merge(vocab: list[str | bytes], vocab_size: int, corpus: str) -> None:
    token_counter = Counter(pretokenize(corpus))

    original_len_vocab = len(vocab)
    for _ in range(original_len_vocab, vocab_size):
        print(f"{token_counter=}")

        pair_counter: Counter[tuple[bytes, bytes]] = Counter()
        for token, count in token_counter.items():
            # NOTE: dict 推导式会把 token 内重复出现的 pair 折叠成一次计数。
            # 本语料任何 token（含中间态）都不含重复相邻 pair，故等价；真语料需改为逐个累加。
            pair_counter.update({pair: count for pair in zip(token, token[1:])})

        max_pair = _find_max_pair_in_counter(pair_counter)
        vocab.append(max_pair[0] + max_pair[1])
        print(f"{max_pair=}")

        # _apply_merge_in_token 在 max_pair 不出现时原样返回，故无需先判断是否命中
        new_token_counter: Counter[tuple[bytes, ...]] = Counter()
        for token, count in token_counter.items():
            new_token_counter[_apply_merge_in_token(max_pair, token)] += count

        token_counter = new_token_counter


if __name__ == "__main__":
    merge(VOCAB, 264, CORPUS)
