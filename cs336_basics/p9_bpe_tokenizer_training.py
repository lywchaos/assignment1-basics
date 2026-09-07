from typing import BinaryIO
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
WordCounts = dict[Word, int]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    for t in split_special_tokens:
        assert isinstance(t, bytes), "Must represent special token as a bytestring"

    pat = regex.compile(
        b"|".join(
            regex.escape(t)
            for t in sorted(
                split_special_tokens,
                key=len,
                reverse=True,
            )
        )
    )

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
            match = pat.search(mini_chunk)
            if match:
                found_at = match.start()
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _prepare_docs(input_path: str | os.PathLike, special_tokens: list[str]) -> list[str]:
    # A example impl for MVP test
    with open(input_path, encoding="utf-8") as file:
        corpus = file.read()

    if not special_tokens:
        return [corpus]
    if any(token == "" for token in special_tokens):
        raise ValueError("special_tokens must not contain empty strings")

    pattern = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return regex.split(pattern, corpus)


def prepare_docs(input_path: str | os.PathLike, special_tokens: list[str]) -> list[str]:
    with open(input_path, "rb") as f:
        split_special_tokens = [t.encode() for t in special_tokens]
        num_chunk = os.cpu_count() or 1
        boundaries = find_chunk_boundaries(f, num_chunk, split_special_tokens)
        split_pattern = "|".join(regex.escape(t) for t in special_tokens)

        ret = []
        for b, e in zip(boundaries, boundaries[1:]):
            f.seek(b)
            chunk = f.read(e - b)
            ret.extend(regex.split(split_pattern, chunk.decode("utf-8")))
        return ret


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
    token_seq_counters = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(pretokenize, doc) for doc in docs]
        for future in as_completed(futures):
            token_seq_counters.append(future.result())
    token_seq_counter = merge_counter(token_seq_counters)

    merges: list[Pair] = []
    for _ in range(len_init_vocab, vocab_size):
        # TODO: Cache pair counts and pair-to-word IDs, then update only words affected by each merge.
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
