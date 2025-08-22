import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    special_tokens = special_tokens or []
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i, special_token in enumerate(special_tokens):
        vocab[i + 256] = special_token.encode()
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


def worker(chunk: str) -> list[str]:
    return pre_tokenize(chunk)


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

    tokens = []
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(
            f, num_processes, b"<|endoftext|>"
        )  # 先硬编码 special_token

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # 移除特殊标记，避免它们被BPE分割
                chunk_parts = split_by_special_token(chunk, special_tokens)
                for part in chunk_parts:
                    if part.strip():  # 跳过空字符串
                        futures.append(executor.submit(worker, part))
            for future in as_completed(futures):
                tokens.extend(future.result())

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
