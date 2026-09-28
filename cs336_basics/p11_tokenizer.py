from __future__ import annotations

import json
from collections.abc import Iterable, Iterator


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
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError
