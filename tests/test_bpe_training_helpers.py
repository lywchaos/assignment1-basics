from pathlib import Path

import pytest

from cs336_basics.p9_bpe_tokenizer_training import merge_word, prepare_docs, pretokenize, train


@pytest.mark.parametrize(
    ("word", "pair", "expected"),
    [
        ((b"h", b"e", b"l", b"l", b"o"), (b"h", b"e"), (b"he", b"l", b"l", b"o")),
        ((b"h", b"e", b"l", b"l", b"o"), (b"l", b"o"), (b"h", b"e", b"l", b"lo")),
        ((b"a", b"b"), (b"a", b"b"), (b"ab",)),
        ((b"a", b"a", b"a"), (b"a", b"a"), (b"aa", b"a")),
    ],
)
def test_merge_word_merges_left_to_right_without_dropping_tokens(word, pair, expected):
    assert merge_word(word, pair) == expected


def test_prepare_docs_without_special_tokens(tmp_path: Path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello 世界", encoding="utf-8")

    assert prepare_docs(input_path, []) == ["hello 世界"]


def test_prepare_docs_matches_longest_special_token_first(tmp_path: Path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("a<|eot|>suffixb", encoding="utf-8")

    assert prepare_docs(input_path, ["<|eot|>", "<|eot|>suffix"]) == ["a", "b"]


def test_prepare_docs_rejects_empty_special_token(tmp_path: Path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain empty strings"):
        prepare_docs(input_path, [""])


def test_pretokenize_splits_utf8_into_single_byte_tokens():
    assert pretokenize("é") == {(b"\xc3", b"\xa9"): 1}


def test_train_stops_when_corpus_has_no_pairs(tmp_path: Path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("a", encoding="utf-8")

    vocab, merges = train(input_path, vocab_size=300, special_tokens=[])

    assert len(vocab) == 256
    assert merges == []


def test_train_rejects_vocab_size_smaller_than_initial_vocab(tmp_path: Path):
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="vocab_size must be at least 257"):
        train(input_path, vocab_size=256, special_tokens=["<|endoftext|>"])
