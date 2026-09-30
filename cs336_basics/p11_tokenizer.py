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
        # Provenance note. The artifacts in artifacts/p12_tokenizer_experiments/ were produced
        # with the version of this streaming encoder at commit ca7af3f (the state before the
        # readability refactor of this method); `git show ca7af3f:cs336_basics/p11_tokenizer.py`
        # recovers it. The two versions were verified equivalent (random fuzz plus per-ID
        # comparison on tests/fixtures, 1.27M tokens), so the artifacts stay valid unless that
        # equivalence is later disproven; if it is, the .npy files are regenerated, not patched.
        specials = sorted(self.special_tokens or [], key=len, reverse=True)
        special_re = regex.compile("|".join(map(regex.escape, specials))) if specials else None
        hold = max(len(specials[0]) - 1, 0) if specials else 0

        buffer = ""
        for chunk in iterable:
            buffer += chunk
            safe_cut = len(buffer) - hold

            # 1) Real boundary: the last special whose start no future chunk can extend or divide.
            cut = 0
            if special_re is not None:
                for match in special_re.finditer(buffer):
                    if match.start() >= safe_cut:
                        break
                    cut = match.end()
            yield from self.encode(buffer[:cut])
            buffer, safe_cut = buffer[cut:], safe_cut - cut

            # 2) Fake boundary: safe_len ends an arbitrary slice, so PAT may read past it.
            safe_len = max(safe_cut, 0)
            matches = list(regex.finditer(PAT, buffer[:safe_len]))
            # Keep two matches: a future chunk can extend the last one and, by splitting it in two,
            # turn the current second-to-last into the third-to-last. "Two" = PAT's read-ahead (1
            # char past a match) + one split, see boundary-thinking.md.
            for match in matches[:-2]:
                yield from self._encode_pretoken(to_word(match.group()))
            if len(matches) > 2:
                buffer = buffer[matches[-2].start() :]

        yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
