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
        # 产物出处：artifacts/p12_tokenizer_experiments/ 里的 .npy 由本方法重构前的版本产出，
        # 即提交 ca7af3f 的状态，`git show ca7af3f:cs336_basics/p11_tokenizer.py` 可恢复。
        # 两个版本已验证等价（随机对拍 + fixtures 逐 ID 对比，约 127 万 token），
        # 因此除非该等价性被证伪，产物保持有效；一旦证伪，重跑产物而不是修补 .npy。
        specials = sorted(self.special_tokens or [], key=len, reverse=True)
        special_re = regex.compile("|".join(map(regex.escape, specials))) if specials else None
        hold = max(len(specials[0]) - 1, 0) if specials else 0

        buffer = ""
        for chunk in iterable:
            buffer += chunk
            # 末尾 hold 个字符可能是尚未到齐的 special，PAT 又可能越过 match 末尾再读一个字符
            # 来做判断，所以只有 buffer[:safe_len] 才敢扫描定性，safe_len 不会为负。
            safe_len = max(len(buffer) - hold, 0)

            # 1) 真边界：起点落在 safe_len 之前的 special，后续 chunk 既无法延长它、也无法把它切开。
            # 比它更长的 special 仍可能从更靠后的位置起点开始并吞掉当前内容，所以只看起点。
            settled = 0
            if special_re is not None:
                for match in special_re.finditer(buffer):
                    if match.start() >= safe_len:
                        break
                    settled = match.end()
            yield from self.encode(buffer[:settled])
            buffer = buffer[settled:]
            safe_len = max(len(buffer) - hold, 0)  # 在新 buffer 的坐标系里重新推一遍

            # 2) 假边界：safe_len 之前的内容已经定型，PAT 扫描时最多多读一个字符。
            matches = list(regex.finditer(PAT, buffer[:safe_len]))
            # 保留最后两个 match：后续 chunk 会让最后一个 match 继续延长；一旦这个延长把某个
            # match 切成两个，当前的倒数第二个就会变成倒数第三个，所以倒数第二个也不能定稿。
            # "两个" = PAT 的预读（1 个字符）+ 1 次切分，推导见 boundary-thinking.md。
            for match in matches[:-2]:
                yield from self._encode_pretoken(to_word(match.group()))
            if len(matches) > 2:
                buffer = buffer[matches[-2].start() :]

        yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
