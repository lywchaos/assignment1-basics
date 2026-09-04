import regex
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def prepare_docs(input_path: str | os.PathLike, special_tokens: list[str]) -> list[str]:
    with open(input_path) as f:
        corpus = f.read()
    docs = regex.split("|".join([regex.escape(pat) for pat in special_tokens]), corpus)
    return docs


def pretokenize(doc: str, pat: str) -> dict[bytes, int]:
    ret = {}
    for m in regex.finditer(pat, doc):
        text = m.group()
        token = text.encode()
        token_seq = tuple(b for b in token)
        ret[token_seq] = ret.setdefault(token_seq, 0) + 1
    return ret


def merge_counter(counters: list[dict]) -> dict:
    ret = {}
    for c in counters:
        for k, v in c.items():
            ret[k] = ret.setdefault(k, 0) + v
    return ret


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    ret = {i: st.encode() for i, st in enumerate(special_tokens)}
    cur_len = len(ret)
    for i in range(256):
        ret[cur_len + i] = bytes([i])
    return ret


def build_new_seq(seq: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_seq = []
    i = 0
    while i < len(seq) - 1:
        current_pair = (seq[i], seq[i + 1])
        if current_pair == pair:
            new_seq.append(pair)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return tuple(new_seq)


def apply_merge(
    token_seq_counter: dict[tuple[bytes, ...], int], max_pair: tuple[bytes, bytes]
) -> dict[tuple[bytes, ...], int]:
    ret = {}
    for k, v in token_seq_counter.items():
        pairs = list(zip(k, k[1:]))
        seq = k
        if max_pair in pairs:
            seq = build_new_seq(k, max_pair)
        ret[seq] = ret.setdefault(seq, 0) + v
    return ret


def train(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    docs = prepare_docs(input_path, special_tokens)
    token_seq_counter = merge_counter([pretokenize(doc, PAT) for doc in docs])

    vocab = init_vocab(special_tokens)
    merges = []
    len_init_vocab = len(vocab)
    for _ in range(len_init_vocab, vocab_size):
        # counter pair
        pair_counter = {}
        for k, v in token_seq_counter.items():
            pairs = zip(k, k[1:])
            for p in pairs:
                pair_counter[p] = pair_counter.setdefault(p, 0) + v

        # find max pair
        max_count = max([v for k, v in pair_counter.items()])
        max_pairs = [k for k, v in pair_counter.items() if v == max_count]
        max_pair = max(max_pairs)

        # update result
        vocab[len(vocab)] = bytes([max_pair[0], max_pair[1]])
        merges.append(max_pair)

        # merge in original token_seq_counter
        token_seq_counter = apply_merge(token_seq_counter, max_pair)

    return vocab, merges
