import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import BinaryIO

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
    if desired_num_chunks <= 0:
        raise ValueError("desired_num_chunks must be positive")
    if any(not isinstance(token, bytes) for token in split_special_tokens):
        raise TypeError("split_special_tokens must contain only bytes")
    if any(token == b"" for token in split_special_tokens):
        raise ValueError("split_special_tokens must not contain empty strings")

    # Get total file size in bytes.
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Without delimiters, arbitrary byte boundaries can split a UTF-8 character.
    if not split_special_tokens:
        return [0, file_size]

    pat = regex.compile(b"|".join(regex.escape(token) for token in sorted(split_special_tokens, key=len, reverse=True)))

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    overlap_size = max(len(token) for token in split_special_tokens) - 1

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        scan_position = initial_position
        carry = b""
        file.seek(scan_position)  # Start at boundary guess

        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            search_chunk = carry + mini_chunk
            search_origin = scan_position - len(carry)
            for match in pat.finditer(search_chunk):
                found_at = search_origin + match.start()
                if found_at >= initial_position:
                    chunk_boundaries[bi] = found_at
                    break
            else:
                if overlap_size:
                    carry = search_chunk[-overlap_size:]
                scan_position += len(mini_chunk)
                continue
            break

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _split_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    if any(token == "" for token in special_tokens):
        raise ValueError("special_tokens must not contain empty strings")

    pattern = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return regex.split(pattern, text)


def prepare_docs(input_path: str | os.PathLike, special_tokens: list[str]) -> list[str]:
    with open(input_path, encoding="utf-8") as file:
        return _split_special_tokens(file.read(), special_tokens)


def _pretokenize_chunk(
    input_path: str | os.PathLike,
    begin: int,
    end: int,
    special_tokens: list[str],
) -> WordCounts:
    with open(input_path, "rb") as file:
        file.seek(begin)
        chunk = file.read(end - begin)

    counts: Counter[Word] = Counter()
    for doc in _split_special_tokens(chunk.decode("utf-8"), special_tokens):
        counts.update(pretokenize(doc))
    return dict(counts)


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
    """Merge every non-overlapping occurrence of pair from left to right."""
    merged_word: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged_word.append(word[i] + word[i + 1])
            i += 2
        else:
            merged_word.append(word[i])
            i += 1
    return tuple(merged_word)


def _count_word_pairs(word: Word) -> Counter[Pair]:
    """Count pair occurrences within one current word."""
    return Counter(zip(word, word[1:]))


def _remove_word_from_cache(
    word: Word,
    count: int,
    pair_counts: dict[Pair, int],
    pair_to_words: dict[Pair, set[Word]],
) -> None:
    """Remove one word's weighted pair counts and reverse-index memberships."""
    for pair, occurrences in _count_word_pairs(word).items():
        updated_count = pair_counts[pair] - occurrences * count
        if updated_count < 0:
            raise RuntimeError(f"negative pair count for {pair!r}")
        if updated_count == 0:
            del pair_counts[pair]
        else:
            pair_counts[pair] = updated_count

        words = pair_to_words[pair]
        words.remove(word)
        if not words:
            del pair_to_words[pair]


def _add_word_to_cache(
    word: Word,
    count: int,
    pair_counts: dict[Pair, int],
    pair_to_words: dict[Pair, set[Word]],
) -> None:
    """Add one word's weighted pair counts and reverse-index memberships."""
    for pair, occurrences in _count_word_pairs(word).items():
        pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * count
        pair_to_words.setdefault(pair, set()).add(word)


def apply_merge(
    token_seq_counter: WordCounts,
    max_pair: Pair,
    pair_counts: dict[Pair, int],
    pair_to_words: dict[Pair, set[Word]],
) -> None:
    """Apply one merge while updating only words containing max_pair."""
    affected_words = pair_to_words[max_pair].copy()
    new_word_counts: Counter[Word] = Counter()

    # Remove all old words before adding any new words, so collisions are aggregated cleanly.
    for old_word in affected_words:
        count = token_seq_counter.pop(old_word)
        _remove_word_from_cache(old_word, count, pair_counts, pair_to_words)
        new_word_counts[merge_word(old_word, max_pair)] += count

    for new_word, count in new_word_counts.items():
        token_seq_counter[new_word] = token_seq_counter.get(new_word, 0) + count
        _add_word_to_cache(new_word, count, pair_counts, pair_to_words)


def train(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Pair]]:
    vocab = init_vocab(special_tokens)
    len_init_vocab = len(vocab)
    if vocab_size < len_init_vocab:
        raise ValueError(f"vocab_size must be at least {len_init_vocab}")

    num_workers = os.cpu_count() or 1
    special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_workers, special_tokens_bytes)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _pretokenize_chunk,
                input_path,
                begin,
                end,
                special_tokens,
            )
            for begin, end in zip(boundaries, boundaries[1:])
        ]
        token_seq_counter = merge_counter(future.result() for future in as_completed(futures))

    merges: list[Pair] = []

    pair_counts: Counter[Pair] = Counter()
    pair_to_words: defaultdict[Pair, set[Word]] = defaultdict(set)
    for word, count in token_seq_counter.items():
        _add_word_to_cache(word, count, pair_counts, pair_to_words)

    for _ in range(len_init_vocab, vocab_size):
        if not pair_counts:
            break

        max_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]

        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)

        apply_merge(token_seq_counter, max_pair, pair_counts, pair_to_words)

    return vocab, merges
