import os
from collections import defaultdict
from typing import BinaryIO, TypedDict

import regex as re
from loguru import logger


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: list[str],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    split_special_token_list = [i.encode() for i in split_special_token]
    for special_token in split_special_token_list:
        assert isinstance(
            special_token, bytes
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
            found_special_token = False
            for token in split_special_token_list:
                found_at = mini_chunk.find(token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    found_special_token = True
                    break
            if found_special_token:
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class TrainResult(TypedDict):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


def pre_tokenize(text: bytes) -> list[str]:
    _text = text.decode(errors="ignore")
    return [
        m.group()
        for m in re.finditer(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            _text,
        )
    ]


def process_chunk(
    file: BinaryIO, bundary: tuple[int, int], special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    words_counter = defaultdict(int)
    file.seek(bundary[0])

    chunk = file.read(bundary[1] - bundary[0])
    chunks = re.split(b"|".join([re.escape(t.encode()) for t in special_tokens]), chunk)

    for _chunk in chunks:
        for token in pre_tokenize(_chunk):
            words_counter[tuple([i.encode() for i in token])] += 1
    return words_counter


def get_word_counter(
    path: str,
    special_tokens: list[str],
    chunk_size: int = 1024 * 1024,
) -> dict[str, int]:
    file = open(path, "rb")
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    logger.debug(f"{file_size=}")
    file.seek(0)

    # words_counters = []

    # chunk_nums = file_size // chunk_size
    # chunk_boundaries = find_chunk_boundaries(file, chunk_nums, special_tokens)
    # for i in range(len(chunk_boundaries) - 1):
    #     bundary = (chunk_boundaries[i], chunk_boundaries[i + 1])
    #     words_counter = process_chunk(file, bundary, special_tokens)
    #     words_counters.append(words_counter)

    bundary = (0, file_size)
    return process_chunk(file, bundary, special_tokens)

    # words_counter = process_chunk(file, bundary, special_tokens)
    # words_counters.append(words_counter)

    # ret_words_counter = defaultdict(int)
    # for wc in words_counters:
    # for k, v in wc.items():
    # ret_words_counter[k] += v
    # return ret_words_counter


def init_vocab(special_tokens: list[str] | None) -> dict[int, bytes]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    ind = 256
    for token in special_tokens:
        vocab[ind] = token.encode()
        ind += 1
    return vocab


def get_best_pair(
    words_counter: dict[tuple[bytes, ...], int]
) -> tuple[bytes, bytes] | None:
    pair_counter = defaultdict(int)
    for k, v in words_counter.items():
        if len(k) < 2:
            continue
        for i in range(len(k) - 1):
            pair_counter[k[i : i + 2]] += v
    if len(pair_counter) == 0:
        return None

    max_count = max(pair_counter.values())
    best_pair = max([k for k, v in pair_counter.items() if v == max_count])
    return (best_pair[0], best_pair[1])


def time_to_stop(vocab: dict[int, bytes], vocab_size: int) -> bool:
    if len(vocab) >= vocab_size:
        return True
    return False


def merge(
    best_pair: tuple[bytes, bytes],
    vocab: dict[int, bytes],
    words_counter: dict[tuple[bytes, ...], int],
    merges: list[tuple[bytes, bytes]],
):
    merges.append(best_pair)

    # best_pair = (gpt2_bytes_to_unicode()[ord(best_pair[0].decode())].encode(), gpt2_bytes_to_unicode()[ord(best_pair[1].decode())].encode())
    # merges.append(best_pair)

    len_vocab = len(vocab)
    vocab[len_vocab] = b"".join(best_pair)
    kv_to_append = []
    k_to_delete = []
    for k, v in words_counter.items():
        for i in range(len(k) - 1):
            if tuple(k[i : i + 2]) == best_pair:
                kv_to_append.append((k[:i] + (vocab[len_vocab],) + k[i + 2 :], v))
                k_to_delete.append(k)
    for k in k_to_delete:
        if k in words_counter:
            del words_counter[k]
    for k, v in kv_to_append:
        if k not in words_counter:
            words_counter[k] += v
    return words_counter, merges, vocab


def train_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> TrainResult:
    special_tokens = special_tokens or []

    words_counter = get_word_counter(input_path, special_tokens, chunk_size=1024)
    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    while not time_to_stop(vocab, vocab_size):
        best_pair = get_best_pair(words_counter)
        if best_pair is None:  # 只有在合无可合的时候才会返回 None
            break
        words_counter, merges, vocab = merge(best_pair, vocab, words_counter, merges)
    return TrainResult(vocab=vocab, merges=merges)


if __name__ == "__main__":
    # import json

    # res = train_tokenizer(
    #     input_path="data/TinyStoriesV2-GPT4-valid.txt",
    #     # input_path="data/head_owt_valid.txt",
    #     vocab_size=1000,
    #     special_tokens=["<|endoftext|>"],
    # )
    # res["merges"] = [(m[0].decode(), m[1].decode()) for m in res["merges"]]
    # res['vocab'] = {k: v.decode() for k, v in res['vocab'].items()}
    # with open("tokenizer_res.json", "w") as f:
    #     json.dump(res, f, ensure_ascii=False, indent=4)

    wc2 = get_word_counter(
        path="data/TinyStoriesV2-GPT4-valid.txt",
        special_tokens=["<|endoftext|>"],
        chunk_size=1024 * 64,
    )
    wc1 = get_word_counter(
        path="data/TinyStoriesV2-GPT4-valid.txt",
        special_tokens=["<|endoftext|>"],
        chunk_size=1024 * 1024,
    )
    print(len(wc2.keys()))
    print(len(wc1.keys()))
    assert set(wc2.keys()) == set(wc1.keys())
    assert set(wc2.values()) == set(wc1.values())
