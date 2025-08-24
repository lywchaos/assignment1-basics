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


def merge(
    best_pair: tuple[bytes, bytes],
    token_bytes_counter: list[tuple[tuple[bytes, ...], int]],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    pair_in_token_index: set[int],
    byte_pair_counter: defaultdict[tuple[bytes, bytes], int],
) -> None:
    # 将最佳字节对添加到合并列表
    merges.append(best_pair)

    # 将最佳字节对添加到词汇表
    best_pair_bytes = b"".join(best_pair)
    vocab[len(vocab)] = best_pair_bytes

    # 更新数据结构
    for idx in pair_in_token_index[best_pair]:
        token_bytes, count = token_bytes_counter[idx]
        new_token_bytes = []

        fast_ptr = 0
        slow_ptr = 0
        best_pair_in_new_token_bytes_idxs = []
        while fast_ptr < len(token_bytes):
            if (fast_ptr < len(token_bytes) - 1) and (
                token_bytes[fast_ptr : fast_ptr + 2] == best_pair
            ):
                new_token_bytes.append(best_pair_bytes)
                fast_ptr += 2
                best_pair_in_new_token_bytes_idxs.append(slow_ptr)
            else:
                new_token_bytes.append(token_bytes[fast_ptr])
                fast_ptr += 1
            slow_ptr += 1

        for new_bp_index in best_pair_in_new_token_bytes_idxs:
            byte_pair_counter[best_pair] -= count
            if new_bp_index > 0:  # 左侧有重叠 pair
                if new_token_bytes[new_bp_index - 1] == best_pair_bytes:
                    byte_pair_counter[(best_pair[1], best_pair[0])] -= count
                else:
                    byte_pair_counter[
                        (new_token_bytes[new_bp_index - 1], best_pair[0])
                    ] -= count

                pair_in_token_index[
                    (new_token_bytes[new_bp_index - 1], new_token_bytes[new_bp_index])
                ].add(idx)
                byte_pair_counter[
                    (new_token_bytes[new_bp_index - 1], new_token_bytes[new_bp_index])
                ] += count

            if new_bp_index < len(new_token_bytes) - 1:  # 右侧有重叠 pair
                if new_token_bytes[new_bp_index + 1] == best_pair_bytes:
                    byte_pair_counter[(best_pair[1], best_pair[0])] -= count
                else:
                    byte_pair_counter[
                        (best_pair[1], new_token_bytes[new_bp_index + 1])
                    ] -= count

                pair_in_token_index[
                    (new_token_bytes[new_bp_index], new_token_bytes[new_bp_index + 1])
                ].add(idx)
                byte_pair_counter[
                    (new_token_bytes[new_bp_index], new_token_bytes[new_bp_index + 1])
                ] += count

        token_bytes_counter[idx] = (tuple(new_token_bytes), count)


def test_merge():
    """测试 merge 函数的功能"""
    from collections import defaultdict

    from rich import print

    # 初始化测试数据
    # 模拟三个token: "hello", "world", "help"
    token_bytes_counter = [
        ((b"h", b"e", b"l", b"l", b"o"), 2),  # "hello" 出现2次
        ((b"w", b"o", b"r", b"l", b"d"), 1),  # "world" 出现1次
        ((b"h", b"e", b"l", b"p"), 1),  # "help" 出现1次
    ]

    # 初始化词汇表 (前256个是字节)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    # 构建 pair_in_token_index: 记录每个字节对出现在哪些token中
    pair_in_token_index = defaultdict(set)
    for i, (token_bytes, count) in enumerate(token_bytes_counter):
        for j in range(len(token_bytes) - 1):
            pair = (token_bytes[j], token_bytes[j + 1])
            pair_in_token_index[pair].add(i)

    # 构建 byte_pair_counter: 统计字节对频率
    byte_pair_counter = defaultdict(int)
    for token_bytes, count in token_bytes_counter:
        for j in range(len(token_bytes) - 1):
            pair = (token_bytes[j], token_bytes[j + 1])
            byte_pair_counter[pair] += count

    # 要合并的最佳字节对: (b'e', b'l') 出现3次 (hello中2次, help中1次)
    best_pair = (b"e", b"l")

    print("合并前:")
    print("token_bytes_counter:", token_bytes_counter)
    print("byte_pair_counter:", dict(byte_pair_counter))
    print("pair_in_token_index:", dict(pair_in_token_index))
    print("vocab size:", len(vocab))

    # 执行合并
    merge(
        best_pair,
        token_bytes_counter,
        vocab,
        merges,
        pair_in_token_index,
        byte_pair_counter,
    )

    print("\n合并后:")
    print("token_bytes_counter:", token_bytes_counter)
    print("byte_pair_counter:", dict(byte_pair_counter))
    print("pair_in_token_index:", dict(pair_in_token_index))
    print("vocab size:", len(vocab))
    print("new vocab entry:", vocab[256])  # 新合并的token
    print("merges:", merges)


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
        )  # FIXME: 先硬编码 special_token, maybe 后续移除

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
    pair_in_token_index = defaultdict(set)  # 用来记录 pair 出现在哪个 token 里
    for i, t in enumerate(tokens):
        token_bytes_counter[tuple([bytes([i]) for i in t.encode()])] += 1
    token_bytes_counter = [(k, v) for k, v in token_bytes_counter.items()]
    for i, (token_bytes, count) in enumerate(token_bytes_counter):
        for byte1, byte2 in zip(token_bytes, token_bytes[1:]):
            pair_in_token_index[(byte1, byte2)].add(i)
    # 统计字节对出现的次数
    byte_pair_counter = defaultdict(int)
    for token_bytes, count in token_bytes_counter:
        for byte1, byte2 in zip(token_bytes, token_bytes[1:]):
            byte_pair_counter[(byte1, byte2)] += count

    for _ in range(nr_merges):
        # 找到出现次数最多的字节对
        max_value = max(byte_pair_counter.values())
        best_pairs = [k for k, v in byte_pair_counter.items() if v == max_value]
        best_pair = max(best_pairs)

        # 合并，并更新各数据结构
        merge(
            best_pair,
            token_bytes_counter,
            vocab,
            merges,
            pair_in_token_index,
            byte_pair_counter,
        )

    return vocab, merges
