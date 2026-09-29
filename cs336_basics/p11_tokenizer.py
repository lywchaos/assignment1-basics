from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import regex

from cs336_basics.p9_bpe_tokenizer_training import (
    PAT,
    Word,
    iter_pretokens,
    merge_word,
    split_special_tokens,
    to_word,
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        next_id = max(self.vocab, default=-1) + 1
        for special_token in special_tokens or []:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[next_id] = token_bytes
                next_id += 1

        self.token_to_id: dict[bytes, int] = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        # Inverse of the byte-to-unicode map the p9 serializers use for vocab.json/merges.txt.
        byte_values = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
        unicode_codepoints = list(byte_values)
        next_codepoint = 0
        for byte_value in range(256):
            if byte_value not in byte_values:
                byte_values.append(byte_value)
                unicode_codepoints.append(256 + next_codepoint)
                next_codepoint += 1
        byte_decoder = {
            chr(codepoint): byte_value for byte_value, codepoint in zip(byte_values, unicode_codepoints, strict=True)
        }

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            serialized_vocab: dict[str, int] = json.load(vocab_file)
        vocab = {
            token_id: bytes(byte_decoder[character] for character in serialized_token)
            for serialized_token, token_id in serialized_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                fields = line.split()
                if len(fields) != 2:
                    continue
                left, right = fields
                merges.append(
                    (
                        bytes(byte_decoder[character] for character in left),
                        bytes(byte_decoder[character] for character in right),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        special_tokens = self.special_tokens or []
        special_set = set(special_tokens)

        ret: list[int] = []
        for part in split_special_tokens(text, special_tokens):
            if not part:
                continue
            if part in special_set:
                ret.append(self.token_to_id[part.encode("utf-8")])
                continue
            for word in iter_pretokens(part):
                ret.extend(self._encode_pretoken(word))
        return ret

    def _encode_pretoken(self, word: Word) -> list[int]:
        while True:
            pair = min(
                (candidate for candidate in zip(word, word[1:]) if candidate in self.merge_ranks),
                key=self.merge_ranks.__getitem__,
                default=None,
            )
            if pair is None:
                break
            word = merge_word(word, pair)
        return [self.token_to_id[piece] for piece in word]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        special_tokens = self.special_tokens or []
        max_special_len = max((len(token) for token in special_tokens), default=0)
        hold = max(max_special_len - 1, 0)

        buffer = ""
        for chunk in iterable:
            buffer += chunk

            # 1) Emit the prefix ending at the last special no future chunk can extend.
            safe_cut = len(buffer) - hold
            parts = split_special_tokens(buffer, special_tokens)
            pos = 0
            finalized_index = -1
            for index, part in enumerate(parts):
                if index % 2 == 1 and pos < safe_cut:
                    finalized_index = index
                pos += len(part)

            emitted = 0
            for index, part in enumerate(parts[: finalized_index + 1]):
                if index % 2 == 1:
                    yield self.token_to_id[part.encode("utf-8")]
                else:
                    for word in iter_pretokens(part):
                        yield from self._encode_pretoken(word)
                emitted += len(part)
            buffer = buffer[emitted:]

            # 2) Emit pre-tokens before the hold suffix, keeping the last matches for the next chunk.
            safe_len = len(buffer) - hold
            if safe_len <= 0:
                continue
            # Keep two matches: a contraction such as "'re" can be split by the slice into a
            # separate "'" match plus the remaining letters, and only the letters would be last.
            pending = []
            for match in regex.finditer(PAT, buffer[:safe_len]):
                if len(pending) == 2:
                    yield from self._encode_pretoken(to_word(pending.pop(0).group()))
                pending.append(match)
            if pending:
                buffer = buffer[pending[0].start() :]

        yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
