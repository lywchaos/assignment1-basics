"""预分词并发与内存限制的回归测试。"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import pytest

from cs336_basics import p9_bpe_tokenizer_training as bpe
from cs336_basics import p9_train_bpe_tinystories as runner

SPECIAL = "<|endoftext|>"


class _TrackedFuture(Future[bpe.WordCounts]):
    """在调度器消费结果时递减 in-flight 计数。"""

    def __init__(self, executor: _FakeExecutor) -> None:
        super().__init__()
        self._executor = executor

    def result(self, timeout: float | None = None) -> bpe.WordCounts:
        self._executor.outstanding -= 1
        return super().result(timeout)


class _FakeExecutor:
    """同步执行任务，并记录调度器同时在手（已 submit 但未取结果）的任务数。"""

    def __init__(self, *, max_workers: int) -> None:
        self.max_workers = max_workers
        self.outstanding = 0
        self.peak_outstanding = 0
        self.submitted = 0

    def __enter__(self) -> _FakeExecutor:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def submit(self, fn: Callable[..., bpe.WordCounts], *args: object, **kwargs: object) -> _TrackedFuture:
        self.submitted += 1
        self.outstanding += 1
        self.peak_outstanding = max(self.peak_outstanding, self.outstanding)
        future = _TrackedFuture(self)
        future.set_result(fn(*args, **kwargs))
        return future


def _write_corpus(path: Path, repeats: int) -> None:
    path.write_text(("hello world " + SPECIAL) * repeats, encoding="utf-8")


@pytest.mark.parametrize(("num_workers", "expected_workers"), [(2, 2), (10, 6)])
def test_pretokenization_bounds_in_flight_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
    expected_workers: int,
) -> None:
    input_path = tmp_path / "corpus.txt"
    _write_corpus(input_path, repeats=6)
    size = input_path.stat().st_size
    chunk_size = size // 6
    boundaries = [index * chunk_size for index in range(6)] + [size]
    monkeypatch.setattr(bpe, "find_chunk_boundaries", lambda *_args, **_kwargs: boundaries)

    executors: list[_FakeExecutor] = []

    def executor_factory(*, max_workers: int) -> _FakeExecutor:
        executor = _FakeExecutor(max_workers=max_workers)
        executors.append(executor)
        return executor

    monkeypatch.setattr(bpe, "ProcessPoolExecutor", executor_factory)
    messages: list[str] = []
    bpe.train(
        input_path,
        257,
        [SPECIAL],
        show_progress=False,
        status_callback=messages.append,
        num_workers=num_workers,
        max_chunk_mib=1,
    )

    assert len(executors) == 1
    executor = executors[0]
    # 同时运行的 worker 数取 num_workers 与 chunk 数的较小值。
    assert executor.max_workers == expected_workers
    # 每个 chunk 都被处理，且调度器在手任务数从不超过 worker 上限。
    assert executor.submitted == 6
    assert executor.peak_outstanding <= expected_workers
    assert any(f"6 chunks with {expected_workers} workers" in message for message in messages)


def test_boundaries_collapse_when_delimiter_is_absent(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    input_path.write_text("no delimiter here " * 100, encoding="utf-8")

    with input_path.open("rb") as file:
        boundaries = bpe.find_chunk_boundaries(file, 8, [SPECIAL.encode()])

    assert boundaries == [0, input_path.stat().st_size]


def test_chunked_pretokenization_matches_whole_corpus(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    _write_corpus(input_path, repeats=200)
    special_tokens = [SPECIAL]

    with input_path.open("rb") as file:
        boundaries = bpe.find_chunk_boundaries(file, 3, [SPECIAL.encode()])
    assert len(boundaries) == 4  # [0, b1, b2, EOF] -> 3 chunks

    chunked = bpe.merge_counter(
        bpe._pretokenize_chunk(input_path, begin, end, special_tokens) for begin, end in zip(boundaries, boundaries[1:])
    )
    whole = bpe.merge_counter(
        bpe.pretokenize(doc)
        for doc in bpe._split_special_tokens(input_path.read_text(encoding="utf-8"), special_tokens)
    )
    assert dict(chunked) == whole


def test_train_rejects_non_positive_limits(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    _write_corpus(input_path, repeats=1)

    with pytest.raises(ValueError, match="num_workers must be positive"):
        bpe.train(input_path, 257, [SPECIAL], num_workers=0)
    with pytest.raises(ValueError, match="num_workers must be positive"):
        bpe.train(input_path, 257, [SPECIAL], num_workers=-1)
    with pytest.raises(ValueError, match="max_chunk_mib must be positive"):
        bpe.train(input_path, 257, [SPECIAL], max_chunk_mib=0)
    with pytest.raises(ValueError, match="max_chunk_mib must be positive"):
        bpe.train(input_path, 257, [SPECIAL], max_chunk_mib=-5)


def test_runner_rejects_non_positive_limits(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    _write_corpus(input_path, repeats=1)

    with pytest.raises(ValueError, match="num_workers must be positive"):
        runner.train_tinystories(
            input_path=input_path,
            output_dir=tmp_path / "out-workers",
            vocab_size=257,
            num_workers=0,
            show_progress=False,
        )
    with pytest.raises(ValueError, match="max_chunk_mib must be positive"):
        runner.train_tinystories(
            input_path=input_path,
            output_dir=tmp_path / "out-chunks",
            vocab_size=257,
            max_chunk_mib=0,
            show_progress=False,
        )


def test_cli_parses_resource_limits() -> None:
    parser = runner._build_parser()
    defaults = parser.parse_args([])
    assert defaults.num_workers == bpe.DEFAULT_NUM_WORKERS
    assert defaults.max_chunk_mib == bpe.DEFAULT_MAX_CHUNK_MIB

    parsed = parser.parse_args(["--num-workers", "2", "--max-chunk-mib", "32"])
    assert parsed.num_workers == 2
    assert parsed.max_chunk_mib == 32

    with pytest.raises(SystemExit):
        parser.parse_args(["--num-workers", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-chunk-mib", "0"])


def test_worklog_records_pretokenization_limits(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    _write_corpus(input_path, repeats=2)
    output_dir = tmp_path / "out"

    worklog = runner.train_tinystories(
        input_path=input_path,
        output_dir=output_dir,
        vocab_size=257,
        num_workers=1,
        max_chunk_mib=1,
        show_progress=False,
    )

    assert worklog["training"]["pretokenization"] == {"num_workers": 1, "max_chunk_mib": 1}
    log_text = (output_dir / runner.TRAIN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "Pre-tokenization limits: 1 workers, 1 MiB target chunk size" in log_text


def test_train_runs_with_real_workers_across_chunks(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.txt"
    sentence = "The quick brown fox jumps over the lazy dog. "
    input_path.write_text((sentence + SPECIAL) * 20_000, encoding="utf-8")

    messages: list[str] = []
    vocab, merges = bpe.train(
        input_path,
        260,
        [SPECIAL],
        show_progress=False,
        status_callback=messages.append,
        num_workers=2,
        max_chunk_mib=1,
    )

    assert len(vocab) == 260
    assert len(merges) == 3
    assert any("2 chunks with 2 workers" in message for message in messages)
