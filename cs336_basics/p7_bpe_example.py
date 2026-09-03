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


if __name__ == "__main__":
    print(pretokenize(CORPUS))
