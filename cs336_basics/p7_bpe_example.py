from rich import print

CORPUS = """low low low low low lower lower widest widest widest newest newest newest newest newest newest"""

VOCAB = [
    "<|endoftext|>",
    *[bytes([i]) for i in range(256)],
]


def pretokenize(corpus: str) -> dict[tuple[bytes, ...], int]:
    chunks = corpus.split(" ")
    counter = {}
    for c in chunks:
        counter[c] = counter.setdefault(c, 0) + 1
    ret = {}
    for w, c in counter.items():
        k = tuple([bytes([ord(i)]) for i in w])
        ret[k] = c
    return ret


if __name__ == "__main__":
    print(pretokenize(CORPUS))
