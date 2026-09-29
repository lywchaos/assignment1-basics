"""P12 tokenizer 实验脚本的回归测试：uint16 .npy 产物与进度输出。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cs336_basics.p11_tokenizer import Tokenizer
from cs336_basics.p12_tokenizer_experiments import encode_dataset


def _tokenizer() -> Tokenizer:
    vocab = {0: b"<|endoftext|>"} | {2 + index: bytes([index]) for index in range(256)}
    return Tokenizer(vocab, [], ["<|endoftext|>"])


def test_encode_dataset_writes_uint16_npy(tmp_path: Path) -> None:
    text = "hello world\n<|endoftext|>abc" * 1000
    input_path = tmp_path / "in.txt"
    output_path = tmp_path / "out.npy"
    input_path.write_text(text, encoding="utf-8")

    tokenizer = _tokenizer()
    count = encode_dataset(tokenizer, input_path, output_path, progress_interval=0.0)

    expected = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    array = np.load(output_path)
    assert count == len(expected)
    assert array.dtype == np.uint16
    assert array.shape == expected.shape
    assert np.array_equal(array, expected)
    # header 在结束时才回填，mmap 读取也要一致
    assert np.array_equal(np.load(output_path, mmap_mode="r")[:], expected)


def test_encode_dataset_handles_empty_input(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.txt"
    input_path.write_text("", encoding="utf-8")

    assert encode_dataset(_tokenizer(), input_path, tmp_path / "empty.npy", progress_interval=0.0) == 0
    array = np.load(tmp_path / "empty.npy")
    assert array.shape == (0,)
    assert array.dtype == np.uint16


def test_encode_dataset_reports_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "in.txt"
    input_path.write_text("hello " * 10_000, encoding="utf-8")

    encode_dataset(_tokenizer(), input_path, tmp_path / "out.npy", label="toy", progress_interval=0.0)

    captured = capsys.readouterr()
    assert "(d) toy: encoding" in captured.out
    assert "(d) toy: 100.0% " in captured.out
    assert "ETA 0s" in captured.out
