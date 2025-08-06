import os
from collections import defaultdict
import regex as re
from loguru import logger
from typer import Typer
from typing import TypedDict, Optional
from io import BytesIO

cli = Typer(pretty_exceptions_enable=False)

def vocabulary_init() -> dict[int, bytes]:
    return {
        i: bytes(i)
        for i in range(256)
    }


def pre_tokenize(text: str) -> list[str]:
    return [
        m.group()
        for m in re.finditer(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""", text)  # https://github.com/openai/tiktoken/pull/234
    ]


def sublist_contains(mainlist: list[bytes], sublist: list[bytes]) -> int:
    if len(mainlist) < len(sublist):
        return -1
    for i in range(len(mainlist) - len(sublist) + 1):
        if mainlist[i:i+len(sublist)] == sublist:
            return i
    return -1


class TrainResult(TypedDict):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


def train(
        input_path: str,
        vocab_size: int,
        special_tokens: Optional[list[str]] = None,
) -> TrainResult:
    special_tokens = special_tokens or ["<|endoftext|"]  # hard code

    # read corpus
    file: BytesIO = open(input_path, "rb")
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    logger.info(f"{file_size=}")
    file.seek(0)

    # 按 chunk 读取，并在 chunk 粒度进行预分词和词频统计
    chunk_size = 4096  # hard code
    token_counts = defaultdict(int)
    for _ in range(0, file_size, chunk_size):
        chunk = file.read(chunk_size)
        pre_tokens = pre_tokenize(chunk.decode("utf-8", errors="ignore"))
        for t in pre_tokens:
            _t = tuple(bytes(byte, "utf-8") for byte in t)
            token_counts[_t] += 1

    # 初始化词表
    vocab = vocabulary_init()
    _vsize = len(vocab)
    for st in special_tokens:
        vocab[_vsize] = st.encode("utf-8")
        _vsize += 1
    logger.info(f"{_vsize=}")

    # 计算合并
    merges = []
    while len(merges) + len(vocab) < vocab_size:
        # 取最频繁的 bigram
        pair = defaultdict(int)
        for token, count in token_counts.items():
            for i in range(len(token) - 1):
                pair[token[i:i + 2]] += count
        best_pair = max(pair, key=pair.get)

        # 更新词表
        merges.append(b"".join(best_pair))
        vocab[_vsize] = b"".join(best_pair)
        _vsize += 1

        # 因为词表更新，所以词频的 key 需要更新
        k_dels = []
        kv_append = []
        for k, v in token_counts.items():
            if (ind := sublist_contains(k, best_pair)) != -1:
                kv_append.append((k[:ind] + tuple([b"".join(best_pair)]) + k[ind + len(best_pair):], v))
                logger.info(f"append: {k} -> {k[:ind] + tuple([b''.join(best_pair)]) + k[ind + len(best_pair):]}")
                k_dels.append(k)
        for k_del in k_dels:
            del token_counts[k_del]
        for k, v in kv_append:
            token_counts[k] += v

        logger.info(f"{_vsize=}")
        logger.info(f"{merges=}")
        logger.info(f"{len(token_counts)=}")
        for k, v in token_counts.items():
            logger.info(f"{k=}, {v=}")
        input()


@cli.command()
def test():
    train(
        input_path="data/head_owt_valid.txt",
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )


if __name__ == "__main__":
    cli()
