"""Tokenizer.encode_iterable 的流式边界回归测试。

覆盖两类曾经出错的边界：

- contraction（``'s``/``'t``/``'d``/``'m``/``'ll``/``'ve``/``'re``）被 chunk 边界切开时，
  slice 里会出现 ``"'"`` 和一个残留字母两个 match，只保留最后一个 match 会把 ``"'"`` 提前 emit；
- special token 被 chunk 切开、以及空白 run 跨 chunk 的分段问题。
"""

from __future__ import annotations

import itertools

import pytest

from cs336_basics.p11_tokenizer import Tokenizer

CONTRACTIONS = ("s", "t", "d", "m", "ll", "ve", "re")

MERGED_TOKENS = (
    b"'r",
    b"'re",
    b"'l",
    b"'ll",
    b"'v",
    b"'ve",
    b"'s",
    b"'t",
    b"'d",
    b"'m",
    b"  ",
    b"\n\n",
)
MERGES = [
    (b"'", b"r"),
    (b"'r", b"e"),
    (b"'", b"l"),
    (b"'l", b"l"),
    (b"'", b"v"),
    (b"'v", b"e"),
    (b"'", b"s"),
    (b"'", b"t"),
    (b"'", b"d"),
    (b"'", b"m"),
    (b" ", b" "),
    (b"\n", b"\n"),
]


def _tokenizer(special_tokens: list[str] | None = None) -> Tokenizer:
    """A tiny byte-level tokenizer whose merges make split contractions change token IDs."""
    vocab = {index: bytes([index]) for index in range(256)}
    vocab.update({256 + offset: token for offset, token in enumerate(MERGED_TOKENS)})
    return Tokenizer(vocab, list(MERGES), special_tokens)


def _assert_streaming_matches(text: str, tokenizer: Tokenizer, chunks: list[str]) -> None:
    assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text), (text, chunks)


@pytest.mark.parametrize("suffix", CONTRACTIONS)
def test_contraction_never_splits_across_chunks(suffix: str) -> None:
    tokenizer = _tokenizer()
    text = "a'" + suffix + "b"
    for cut in range(len(text) + 1):
        _assert_streaming_matches(text, tokenizer, [text[:cut], text[cut:]])


def test_contraction_regression_case() -> None:
    # 真实语料里出事的是 "You're"：slice 切在 "'r" 之后，旧实现会 emit "'"。
    tokenizer = _tokenizer()
    text = "You're welcome"
    for chunks in (
        ["You'r", "e welcome"],
        ["You'", "re welcome"],
        ["You", "'re welcome"],
        ["You'r", "e", " welcome"],
        ["You", "'r", "e welcome"],
    ):
        _assert_streaming_matches(text, tokenizer, chunks)


def test_streaming_matches_encode_on_exhaustive_small_inputs() -> None:
    tokenizer = _tokenizer()
    for length in range(5):
        for chars in itertools.product("a'lrve", repeat=length):
            text = "".join(chars)
            for cut in range(len(text) + 1):
                _assert_streaming_matches(text, tokenizer, [text[:cut], text[cut:]])


def test_streaming_matches_encode_on_exhaustive_small_inputs_with_special_tokens() -> None:
    tokenizer = _tokenizer(["<|x|>"])
    for length in range(5):
        for chars in itertools.product("a'r<|>x", repeat=length):
            text = "".join(chars)
            for cut in range(len(text) + 1):
                _assert_streaming_matches(text, tokenizer, [text[:cut], text[cut:]])


@pytest.mark.parametrize("text", ["", "a'b", "a  \n\n\nb", "a\n\n\n\nb", "  ", "\n\n\n"])
def test_streaming_holds_whitespace_runs_and_empty_text(text: str) -> None:
    tokenizer = _tokenizer()
    for cut in range(len(text) + 1):
        _assert_streaming_matches(text, tokenizer, [text[:cut], text[cut:]])
    _assert_streaming_matches(text, tokenizer, [""] + list(text) + [""])


@pytest.mark.parametrize("text", ["a<|x|>b", "x<|x|>y", "x<|x", "<|x|><|x|>", "x<|"])
def test_streaming_special_token_boundaries(text: str) -> None:
    tokenizer = _tokenizer(["<|x|>"])
    for cut in range(len(text) + 1):
        _assert_streaming_matches(text, tokenizer, [text[:cut], text[cut:]])
