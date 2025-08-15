from collections import defaultdict
from typing import TypedDict

import regex as re


class TrainResult(TypedDict):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]


def end_good(chunk: bytes, special_tokens: list[str] | None) -> bool:
    special_tokens = special_tokens or []
    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")
        if len(special_bytes) > len(chunk):
            continue
        for i in range(len(special_bytes)):
            special_sub_chunk = chunk[: i + 1]
            chunk_sub_chunk = chunk[-(len(special_sub_chunk)) :]
            if special_sub_chunk == chunk_sub_chunk:
                return False
    return True


def pre_tokenize(text: bytes) -> list[bytes]:
    _text = text.decode(errors="ignore")
    return [
        m.group()
        for m in re.finditer(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            _text,
        )  # https://github.com/openai/tiktoken/pull/234
    ]


def process_chunk(
    chunk: bytes,
    special_tokens: list[str] | None,
    index: int,
    words_counter: dict[tuple[bytes, ...], int],
):
    special_tokens = special_tokens or []
    for token in special_tokens:
        chunk = chunk.replace(token.encode("utf-8"), b"")
    for token in pre_tokenize(chunk):
        words_counter[tuple([i.encode() for i in token])] += 1
    index += len(chunk)
    return index, words_counter


def get_words_counter(
    input_path: str, special_tokens: list[str] | None
) -> dict[list[bytes], int]:
    index = 0
    chunk_size = 4096
    corpus = open(input_path, "rb")
    words_counter = defaultdict(int)
    special_tokens = special_tokens or []
    mini_chunk_size = 10

    while True:
        corpus.seek(index)
        chunk = corpus.read(chunk_size)
        if not chunk:
            break
        if end_good(chunk, special_tokens):
            index, words_counter = process_chunk(
                chunk, special_tokens, index, words_counter
            )
        else:
            forward_bytes = b""
            already_end = False
            while True:
                read_forward = corpus.read(mini_chunk_size)
                if not read_forward:
                    already_end = True
                    break
                forward_bytes += read_forward
                if end_good(chunk + forward_bytes, special_tokens):
                    index, words_counter = process_chunk(
                        forward_bytes, special_tokens, index, words_counter
                    )
                    break
            if already_end:
                break

    return words_counter


def init_vocab(special_tokens: list[str] | None) -> dict[int, bytes]:
    vocab = {i: bytes(i) for i in range(256)}
    ind = 256
    for token in special_tokens:
        vocab[ind] = token.encode()
        ind += 1
    return vocab


def time_to_stop(vocab: dict[int, bytes], vocab_size: int) -> bool:
    if len(vocab) >= vocab_size:
        return True
    return False


def get_best_pair(
    words_counter: dict[tuple[bytes, ...], int]
) -> tuple[bytes, bytes] | None:
    pair_counter = defaultdict(int)
    for k, v in words_counter.items():
        if len(k) < 2:
            continue
        for i in range(len(k) - 2):
            pair_counter[k[i : i + 2]] += v
    if len(pair_counter) == 0:
        return None
    best_pair = max(pair_counter, key=pair_counter.get)
    return (best_pair[0], best_pair[1])


def merge(
    best_pair: tuple[bytes, bytes],
    vocab: dict[int, bytes],
    words_counter: dict[tuple[bytes, ...], int],
    merges: list[tuple[bytes, bytes]],
):
    merges.append(best_pair)
    len_vocab = len(vocab)
    vocab[len_vocab] = b"".join(best_pair)
    kv_to_append = []
    k_to_delete = []
    for k, v in words_counter.items():
        for i in range(len(k) - 2):
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

    words_counter = get_words_counter(input_path, special_tokens)
    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    while not time_to_stop(vocab, vocab_size):
        best_pair = get_best_pair(words_counter)
        if best_pair is None:  # 只有在合无可合的时候才会返回 None
            break
        words_counter, merges, vocab = merge(best_pair, vocab, words_counter, merges)
    return TrainResult(vocab=vocab, merges=merges)


if __name__ == "__main__":
    import json

    res = train_tokenizer(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        # input_path="data/head_owt_valid.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    res["merges"] = [(m[0].decode(), m[1].decode()) for m in res["merges"]]
    res["vocab"] = {k: v.decode() for k, v in res["vocab"].items()}
    with open("tokenizer_res.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
