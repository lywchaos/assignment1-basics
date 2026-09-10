"""Run the TinyStories BPE experiment and write inspectable artifacts.

The tokenizer artifacts use the GPT-2 byte-level text format:

* ``vocab.json`` maps the GPT-2 printable representation of each byte string to
  its integer token ID.
* ``merges.txt`` contains one merge pair per line, in creation order (without a header, matching this assignment's loader).

The byte-to-unicode representation is lossless, so this format can represent
vocabulary entries that are not valid UTF-8 on their own.  ``worklog.json`` is
an additional experiment record; it is not part of the tokenizer format.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import platform
import pstats
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, UTC
from importlib import metadata
from pathlib import Path
from types import FrameType
from typing import Any, cast

import psutil
from loguru import logger
from tqdm.auto import tqdm

from cs336_basics.p9_bpe_tokenizer_training import DEFAULT_MAX_CHUNK_MIB, DEFAULT_NUM_WORKERS, train

Merge = tuple[bytes, bytes]

DEFAULT_INPUT_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
DEFAULT_OUTPUT_DIR = Path("artifacts/p9_tinystories")
DEFAULT_SPECIAL_TOKENS = ("<|endoftext|>",)
VOCAB_FILENAME = "vocab.json"
MERGES_FILENAME = "merges.txt"
WORKLOG_FILENAME = "worklog.json"
PROFILE_FILENAME = "train.cprof"
PROFILE_SUMMARY_FILENAME = "train.cprof.txt"
PY_SPY_FILENAME = "train.py-spy.svg"
TRAIN_LOG_FILENAME = "train.log"
CHILD_TERMINATE_TIMEOUT_SECONDS = 2.0
CHILD_KILL_TIMEOUT_SECONDS = 2.0


class _SignalExit(SystemExit):
    """Carry the terminating signal while preserving the conventional exit code."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(128 + signum)


class _ProcessTreeSignalHandler:
    """Stop descendant processes before turning termination signals into exceptions."""

    def __init__(
        self,
        terminate_timeout_seconds: float = CHILD_TERMINATE_TIMEOUT_SECONDS,
        kill_timeout_seconds: float = CHILD_KILL_TIMEOUT_SECONDS,
    ) -> None:
        self._owner_pid = os.getpid()
        self._terminate_timeout_seconds = terminate_timeout_seconds
        self._kill_timeout_seconds = kill_timeout_seconds
        self._previous_handlers: dict[signal.Signals, Any] = {}
        self._handling_signal = False

    def __enter__(self) -> _ProcessTreeSignalHandler:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal handlers can only be installed from the main thread")

        handled_signals = [signal.SIGINT]
        if hasattr(signal, "SIGTERM"):
            handled_signals.append(signal.SIGTERM)

        try:
            for signum in handled_signals:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self)
        except BaseException:
            self._restore_handlers()
            raise
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self._restore_handlers()

    def _restore_handlers(self) -> None:
        for signum, previous_handler in self._previous_handlers.items():
            signal.signal(signum, previous_handler)
        self._previous_handlers.clear()

    def _forward_signal_from_forked_child(self, signum: int) -> None:
        # A fork-based ProcessPoolExecutor worker inherits this object. Restore
        # normal signal behavior there; only the CLI parent owns tree cleanup.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    @staticmethod
    def _is_resource_tracker(process: psutil.Process) -> bool:
        """Leave multiprocessing's signal-ignoring helper to interpreter shutdown."""
        try:
            return any("multiprocessing.resource_tracker" in argument for argument in process.cmdline())
        except psutil.Error:
            return False

    def stop_descendants(self, *, force: bool = False) -> None:
        """Terminate all worker descendants, escalating to kill when necessary."""
        try:
            descendants = [
                process
                for process in psutil.Process(self._owner_pid).children(recursive=True)
                if not self._is_resource_tracker(process)
            ]
        except psutil.Error as error:
            logger.warning("Could not enumerate child processes during shutdown: {}", error)
            return
        if not descendants:
            return

        action = "Killing" if force else "Terminating"
        logger.warning("{} {} child process(es)", action, len(descendants))
        for process in descendants:
            try:
                process.kill() if force else process.terminate()
            except psutil.NoSuchProcess:
                continue
            except psutil.Error as error:
                logger.warning("Could not stop child process {}: {}", process.pid, error)

        wait_timeout = self._kill_timeout_seconds if force else self._terminate_timeout_seconds
        _, alive = psutil.wait_procs(descendants, timeout=wait_timeout)
        if alive and not force:
            logger.warning("{} child process(es) did not terminate in time; killing them", len(alive))
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    continue
                except psutil.Error as error:
                    logger.warning("Could not kill child process {}: {}", process.pid, error)
            _, alive = psutil.wait_procs(alive, timeout=self._kill_timeout_seconds)

        if alive:
            logger.error("{} child process(es) are still alive after forced shutdown", len(alive))

    def __call__(self, signum: int, _frame: FrameType | None) -> None:
        if os.getpid() != self._owner_pid:
            self._forward_signal_from_forked_child(signum)
            return

        force = self._handling_signal
        self._handling_signal = True
        signal_name = signal.Signals(signum).name
        if force:
            logger.warning("Received {} again; forcing child-process shutdown", signal_name)
        else:
            logger.warning("Received {}; stopping child processes", signal_name)
        self.stop_descendants(force=force)

        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise _SignalExit(signum)


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte-to-unicode mapping used by GPT-2."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))

    unicode_codepoints = byte_values[:]
    next_codepoint = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            unicode_codepoints.append(256 + next_codepoint)
            next_codepoint += 1

    return {byte_value: chr(codepoint) for byte_value, codepoint in zip(byte_values, unicode_codepoints, strict=True)}


def _encode_bytes(token: bytes, byte_to_unicode: Mapping[int, str]) -> str:
    return "".join(byte_to_unicode[byte_value] for byte_value in token)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a text artifact without leaving a partially written final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as file:
            file.write(text)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, text)


def serialize_vocab(vocab: Mapping[int, bytes], path: str | os.PathLike[str]) -> None:
    """Serialize ``vocab`` in the GPT-2/Hugging Face ``vocab.json`` shape."""
    byte_to_unicode = _gpt2_bytes_to_unicode()
    serialized_vocab: dict[str, int] = {}

    for token_id, token_bytes in sorted(vocab.items()):
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"token IDs must be non-negative integers, got {token_id!r}")
        if not isinstance(token_bytes, bytes):
            raise TypeError(f"vocabulary values must be bytes, got {type(token_bytes).__name__}")

        serialized_token = _encode_bytes(token_bytes, byte_to_unicode)
        if serialized_token in serialized_vocab:
            previous_id = serialized_vocab[serialized_token]
            raise ValueError(f"duplicate vocabulary token for IDs {previous_id} and {token_id}")
        serialized_vocab[serialized_token] = token_id

    _atomic_write_text(
        Path(path),
        json.dumps(serialized_vocab, ensure_ascii=False, indent=2) + "\n",
    )


def serialize_merges(merges: Sequence[Merge], path: str | os.PathLike[str]) -> None:
    """Serialize ordered BPE merges in the assignment's ``merges.txt`` shape."""
    byte_to_unicode = _gpt2_bytes_to_unicode()
    lines: list[str] = []

    for merge_index, merge in enumerate(merges):
        if len(merge) != 2:
            raise ValueError(f"merge {merge_index} must contain exactly two tokens")
        left, right = merge
        if not isinstance(left, bytes) or not isinstance(right, bytes):
            raise TypeError(f"merge {merge_index} must contain bytes tokens")
        if not left or not right:
            raise ValueError(f"merge {merge_index} must not contain empty tokens")

        left_text = _encode_bytes(left, byte_to_unicode)
        right_text = _encode_bytes(right, byte_to_unicode)
        lines.append(f"{left_text} {right_text}")

    _atomic_write_text(Path(path), "\n".join(lines) + "\n")


def _summarize_longest_tokens(vocab: Mapping[int, bytes]) -> dict[str, Any]:
    if not vocab:
        return {"byte_length": 0, "count": 0, "tokens": []}

    byte_to_unicode = _gpt2_bytes_to_unicode()
    longest_length = max(len(token_bytes) for token_bytes in vocab.values())
    longest_tokens = []
    for token_id, token_bytes in sorted(vocab.items()):
        if len(token_bytes) != longest_length:
            continue
        longest_tokens.append(
            {
                "id": token_id,
                "byte_length": len(token_bytes),
                "hex": token_bytes.hex(),
                "utf8_with_replacement": token_bytes.decode("utf-8", errors="replace"),
                "gpt2_text": _encode_bytes(token_bytes, byte_to_unicode),
            }
        )

    return {
        "byte_length": longest_length,
        "count": len(longest_tokens),
        "tokens": longest_tokens,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_first_documents(
    input_path: Path,
    output_path: Path,
    max_documents: int,
    document_delimiter: str,
) -> dict[str, Any]:
    """Copy complete delimiter-bounded documents without changing their bytes.

    The delimiter is copied along with each document when it is present.  A
    final unterminated tail is treated as one document, but no delimiter is
    synthesized.  This keeps the subset on the same segmentation contract as
    the full corpus while avoiding a whole-file read.
    """
    if max_documents <= 0:
        raise ValueError("max_documents must be positive")
    if not document_delimiter:
        raise ValueError("document_delimiter must not be empty")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("subset output must differ from the input file")

    delimiter = document_delimiter.encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    documents_copied = 0
    delimited_documents_copied = 0
    bytes_written = 0
    has_unterminated_tail = False
    buffer = bytearray()
    chunk_size = 8 * 1024 * 1024

    try:
        with input_path.open("rb") as source, temporary_path.open("wb") as target:
            stop_copying = False
            while not stop_copying and (chunk := source.read(chunk_size)):
                buffer.extend(chunk)
                while True:
                    delimiter_start = buffer.find(delimiter)
                    if delimiter_start < 0:
                        # Keep enough suffix bytes to detect a delimiter split
                        # across two input chunks.
                        safe_length = max(0, len(buffer) - len(delimiter) + 1)
                        if safe_length:
                            target.write(buffer[:safe_length])
                            bytes_written += safe_length
                            del buffer[:safe_length]
                        break

                    document_end = delimiter_start + len(delimiter)
                    target.write(buffer[:document_end])
                    bytes_written += document_end
                    del buffer[:document_end]
                    documents_copied += 1
                    delimited_documents_copied += 1
                    if documents_copied >= max_documents:
                        stop_copying = True
                        break

            if documents_copied < max_documents and buffer:
                target.write(buffer)
                bytes_written += len(buffer)
                documents_copied += 1
                has_unterminated_tail = True

        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "max_documents": max_documents,
        "documents_copied": documents_copied,
        "delimited_documents_copied": delimited_documents_copied,
        "has_unterminated_tail": has_unterminated_tail,
        "bytes_written": bytes_written,
        "delimiter": document_delimiter,
        "delimiter_bytes": len(delimiter),
        "path": str(output_path),
    }


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _git_metadata(repo_dir: Path) -> dict[str, Any]:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {"available": False, "commit": None, "dirty": None}

    if commit_result.returncode != 0:
        return {"available": False, "commit": None, "dirty": None}
    return {
        "available": True,
        "commit": commit_result.stdout.strip(),
        "dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
    }


def _environment_metadata() -> dict[str, Any]:
    try:
        package_version: str | None = metadata.version("cs336_basics")
    except metadata.PackageNotFoundError:
        package_version = None

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "cs336_basics_version": package_version,
        "git": _git_metadata(Path(__file__).resolve().parents[1]),
    }


class _ResourceSampler:
    """Sample parent plus descendant RSS while the training job is running."""

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("memory sample interval must be positive")
        self._interval_seconds = interval_seconds
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_count = 0
        self._peak_parent_rss = 0
        self._peak_child_rss = 0
        self._peak_total_rss = 0

    def _sample(self) -> None:
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except psutil.Error:
            pass

        seen_pids: set[int] = set()
        parent_rss = 0
        total_rss = 0
        for process in processes:
            if process.pid in seen_pids:
                continue
            seen_pids.add(process.pid)
            try:
                rss = process.memory_info().rss
            except psutil.Error:
                continue
            total_rss += rss
            if process.pid == self._process.pid:
                parent_rss = rss

        child_rss = max(total_rss - parent_rss, 0)
        self._sample_count += 1
        self._peak_parent_rss = max(self._peak_parent_rss, parent_rss)
        self._peak_child_rss = max(self._peak_child_rss, child_rss)
        self._peak_total_rss = max(self._peak_total_rss, total_rss)

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._sample()

    def __enter__(self) -> _ResourceSampler:
        self._sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 2))
        self._sample()

    def summary(self) -> dict[str, Any]:
        return {
            "sample_interval_seconds": self._interval_seconds,
            "samples": self._sample_count,
            "peak_parent_rss_bytes": self._peak_parent_rss,
            "peak_child_rss_bytes": self._peak_child_rss,
            "peak_total_rss_bytes": self._peak_total_rss,
            "peak_total_rss_gib": round(self._peak_total_rss / 2**30, 3),
            "measurement": "sampled sum of RSS for the parent and descendants",
        }


def _write_profile_artifacts(
    profiler: cProfile.Profile,
    profile_path: Path,
    summary_path: Path,
) -> None:
    profiler.dump_stats(str(profile_path))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("cumulative").print_stats(50)
    summary = (
        "# cProfile covers the main process only. ProcessPoolExecutor workers are not included.\n"
        "# Use py-spy --subprocesses for a process-tree profile.\n\n" + stream.getvalue()
    )
    _atomic_write_text(summary_path, summary)


def _tqdm_log_sink(message: Any) -> None:
    tqdm.write(str(message), end="", file=sys.stderr)


def _configure_logging(log_path: Path, show_progress: bool) -> None:
    logger.remove()
    console_sink = _tqdm_log_sink if show_progress else sys.stderr
    logger.add(
        console_sink,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        colorize=False,
    )
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        level="INFO",
        encoding="utf-8",
        mode="w",
    )


def _log_train_status(message: str) -> None:
    logger.info(message)


def _run_train(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    profile_mode: str,
    profile_path: Path | None,
    profile_summary_path: Path | None,
    show_progress: bool,
    num_workers: int,
    max_chunk_mib: int,
) -> tuple[dict[int, bytes], list[Merge]]:
    train_kwargs = {
        "show_progress": show_progress,
        "status_callback": _log_train_status,
        "num_workers": num_workers,
        "max_chunk_mib": max_chunk_mib,
    }
    if profile_mode == "none":
        return train(input_path, vocab_size, special_tokens, **train_kwargs)
    if profile_mode != "cprofile":
        raise ValueError(f"unsupported profile mode: {profile_mode!r}")
    if profile_path is None or profile_summary_path is None:
        raise ValueError("cProfile output paths are required when profiling is enabled")

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        return train(input_path, vocab_size, special_tokens, **train_kwargs)
    finally:
        profiler.disable()
        _write_profile_artifacts(profiler, profile_path, profile_summary_path)
        logger.info("cProfile artifacts written: {} and {}", profile_path, profile_summary_path)


def _recommended_py_spy_command(
    input_path: Path,
    output_dir: Path,
    vocab_size: int,
    special_tokens: Sequence[str],
    hash_input: bool,
    max_documents: int | None,
    num_workers: int,
    max_chunk_mib: int,
) -> str:
    command = [
        "py-spy",
        "record",
        "--subprocesses",
        "--output",
        str(output_dir / PY_SPY_FILENAME),
        "--",
        "uv",
        "run",
        "python",
        "-m",
        "cs336_basics.p9_train_bpe_tinystories",
        "--input-path",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--vocab-size",
        str(vocab_size),
        "--profile",
        "none",
        "--num-workers",
        str(num_workers),
        "--max-chunk-mib",
        str(max_chunk_mib),
    ]
    for special_token in special_tokens:
        command.extend(["--special-token", special_token])
    if hash_input:
        command.append("--hash-input")
    if max_documents is not None:
        command.extend(["--max-documents", str(max_documents)])
    return shlex.join(command)


def train_tinystories(
    input_path: str | os.PathLike[str] = DEFAULT_INPUT_PATH,
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
    vocab_size: int = 10_000,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    profile_mode: str = "none",
    hash_input: bool = False,
    memory_sample_interval: float = 0.25,
    max_documents: int | None = None,
    show_progress: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS,
    max_chunk_mib: int = DEFAULT_MAX_CHUNK_MIB,
) -> dict[str, Any]:
    """Train the tokenizer and return the worklog dictionary."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if profile_mode not in {"none", "cprofile"}:
        raise ValueError(f"unsupported profile mode: {profile_mode!r}")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if max_chunk_mib <= 0:
        raise ValueError("max_chunk_mib must be positive")

    input_path_arg = Path(input_path).expanduser()
    input_path_resolved = input_path_arg.resolve()
    output_dir_arg = Path(output_dir).expanduser()
    output_dir_arg.mkdir(parents=True, exist_ok=True)
    output_dir_resolved = output_dir_arg.resolve()
    log_path = output_dir_resolved / TRAIN_LOG_FILENAME
    _configure_logging(log_path, show_progress)

    special_tokens_list = list(special_tokens)
    document_delimiter = "<|endoftext|>"
    if max_documents is not None and document_delimiter not in special_tokens_list:
        raise ValueError(
            f"max_documents requires {document_delimiter!r} in special_tokens to preserve document boundaries"
        )
    subset_path = (
        output_dir_resolved / f"input_subset_{max_documents}_documents.txt" if max_documents is not None else None
    )
    vocab_path = output_dir_resolved / VOCAB_FILENAME
    merges_path = output_dir_resolved / MERGES_FILENAME
    worklog_path = output_dir_resolved / WORKLOG_FILENAME
    profile_path = output_dir_resolved / PROFILE_FILENAME if profile_mode == "cprofile" else None
    profile_summary_path = output_dir_resolved / PROFILE_SUMMARY_FILENAME if profile_mode == "cprofile" else None
    training_input_path = input_path_resolved
    run_started = time.perf_counter()
    logger.info(
        "Starting BPE tokenizer training | input={} | target vocabulary={} | special tokens={}",
        input_path_resolved,
        vocab_size,
        special_tokens_list,
    )
    logger.info(
        "Pre-tokenization limits: {} workers, {} MiB target chunk size",
        num_workers,
        max_chunk_mib,
    )
    logger.info("Detailed log: {}", log_path)
    if show_progress:
        logger.info("Progress display enabled; stages will update below")
    else:
        logger.info("Progress display disabled")
    started_at = _utc_timestamp()
    artifact_paths: dict[str, str] = {
        "vocab": str(vocab_path),
        "merges": str(merges_path),
        "worklog": str(worklog_path),
        "log": str(log_path),
    }
    if subset_path is not None:
        artifact_paths["input_subset"] = str(subset_path)

    worklog: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "phase": "setup",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "elapsed_seconds": None,
        "input": {
            "source_path": str(input_path_arg),
            "source_resolved_path": str(input_path_resolved),
            "training_path": str(training_input_path),
            "training_resolved_path": str(training_input_path),
            "source_size_bytes": None,
            "training_size_bytes": None,
            "source_modified_at": None,
            "training_modified_at": None,
            "source_sha256": None,
            "training_sha256": None,
            "sha256_requested": hash_input,
            "downsampling": {
                "enabled": max_documents is not None,
                "max_documents": max_documents,
                "document_delimiter": document_delimiter if max_documents is not None else None,
                "subset_path": str(subset_path) if subset_path is not None else None,
                "documents_copied": None,
                "bytes_written": None,
            },
        },
        "training": {
            "vocab_size_requested": vocab_size,
            "special_tokens": special_tokens_list,
            "pretokenization": {
                "num_workers": num_workers,
                "max_chunk_mib": max_chunk_mib,
            },
            "vocab_size_actual": None,
            "merge_count": None,
        },
        "serialization": {
            "format": "GPT-2 byte-level BPE; assignment-compatible headerless merges.txt",
            "vocab_path": str(vocab_path),
            "merges_path": str(merges_path),
            "vocab_size_bytes": None,
            "merges_size_bytes": None,
        },
        "analysis": None,
        "profiling": {
            "requested": profile_mode,
            "cprofile_scope": "main_process" if profile_mode == "cprofile" else None,
            "artifacts": [
                str(profile_path),
                str(profile_summary_path),
            ]
            if profile_path is not None and profile_summary_path is not None
            else [],
            "recommended_py_spy_command": _recommended_py_spy_command(
                input_path_resolved,
                output_dir_resolved,
                vocab_size,
                special_tokens_list,
                hash_input,
                max_documents,
                num_workers,
                max_chunk_mib,
            ),
        },
        "environment": _environment_metadata(),
        "timings_seconds": {},
        "resources": None,
        "artifacts": artifact_paths,
        "error": None,
    }
    timings: dict[str, float] = {}
    input_log = cast(dict[str, Any], worklog["input"])
    serialization_log = cast(dict[str, Any], worklog["serialization"])
    sampler = _ResourceSampler(memory_sample_interval)
    _atomic_write_json(worklog_path, worklog)

    try:
        with sampler:
            setup_started = time.perf_counter()
            if not input_path_resolved.is_file():
                raise FileNotFoundError(f"input file does not exist: {input_path_resolved}")
            source_stat = input_path_resolved.stat()
            input_log["source_size_bytes"] = source_stat.st_size
            input_log["source_modified_at"] = datetime.fromtimestamp(source_stat.st_mtime, UTC).isoformat(
                timespec="seconds"
            )
            logger.info("Input ready: {:.2f} GiB", source_stat.st_size / 2**30)
            timings["input_metadata"] = time.perf_counter() - setup_started

            if max_documents is not None:
                if subset_path is None:
                    raise RuntimeError("internal error: subset path is missing")
                downsampling_started = time.perf_counter()
                downsampling = _copy_first_documents(
                    input_path_resolved,
                    subset_path,
                    max_documents,
                    document_delimiter,
                )
                training_input_path = subset_path
                cast(dict[str, Any], input_log["downsampling"]).update(downsampling)
                logger.info(
                    "Downsampling complete: copied {} documents ({} bytes) to {}",
                    downsampling["documents_copied"],
                    downsampling["bytes_written"],
                    subset_path,
                )
                timings["downsampling"] = time.perf_counter() - downsampling_started

            training_stat = training_input_path.stat()
            input_log["training_path"] = str(training_input_path)
            input_log["training_resolved_path"] = str(training_input_path.resolve())
            input_log["training_size_bytes"] = training_stat.st_size
            input_log["training_modified_at"] = datetime.fromtimestamp(training_stat.st_mtime, UTC).isoformat(
                timespec="seconds"
            )
            logger.info("Training input: {} ({:.2f} GiB)", training_input_path, training_stat.st_size / 2**30)

            if hash_input:
                logger.info("Computing SHA-256 fingerprint for the training input")
                hash_started = time.perf_counter()
                input_log["source_sha256"] = _sha256_file(input_path_resolved)
                if training_input_path == input_path_resolved:
                    input_log["training_sha256"] = input_log["source_sha256"]
                else:
                    input_log["training_sha256"] = _sha256_file(training_input_path)
                logger.info("Training input SHA-256: {}", input_log["training_sha256"])
                timings["input_sha256"] = time.perf_counter() - hash_started

            worklog["phase"] = "training"
            worklog["timings_seconds"] = timings
            _atomic_write_json(worklog_path, worklog)
            logger.info("Starting tokenizer training; this may take a while")
            train_started = time.perf_counter()
            vocab, merges = _run_train(
                training_input_path,
                vocab_size,
                special_tokens_list,
                profile_mode,
                profile_path,
                profile_summary_path,
                show_progress,
                num_workers,
                max_chunk_mib,
            )
            timings["training"] = time.perf_counter() - train_started
            worklog["training"]["vocab_size_actual"] = len(vocab)
            worklog["training"]["merge_count"] = len(merges)
            logger.info(
                "Tokenizer training finished in {:.1f}s: {} merges, vocabulary size {}",
                timings["training"],
                len(merges),
                len(vocab),
            )

            worklog["phase"] = "serialization"
            logger.info("Serializing vocabulary and merge artifacts")
            serialization_started = time.perf_counter()
            serialize_vocab(vocab, vocab_path)
            serialize_merges(merges, merges_path)
            timings["serialization"] = time.perf_counter() - serialization_started
            logger.info("Tokenizer artifacts written to {}", output_dir_resolved)
            serialization_log["vocab_size_bytes"] = vocab_path.stat().st_size
            serialization_log["merges_size_bytes"] = merges_path.stat().st_size

            analysis_started = time.perf_counter()
            worklog["analysis"] = {
                "longest_token": _summarize_longest_tokens(vocab),
                "note": "Token length is measured in raw bytes; gpt2_text is the serialized display form.",
            }
            timings["analysis"] = time.perf_counter() - analysis_started
            logger.info(
                "Longest token: {} bytes ({} matching vocabulary entries)",
                worklog["analysis"]["longest_token"]["byte_length"],
                worklog["analysis"]["longest_token"]["count"],
            )
            worklog["phase"] = "completed"
            worklog["status"] = "completed"
    except BaseException as error:
        failed_phase = worklog["phase"]
        interruption_signal: signal.Signals | None = None
        if isinstance(error, KeyboardInterrupt):
            interruption_signal = signal.SIGINT
        elif isinstance(error, _SignalExit):
            interruption_signal = signal.Signals(error.signum)

        if interruption_signal is not None:
            worklog["phase"] = "interrupted"
            worklog["status"] = "interrupted"
            worklog["error"] = {
                "type": type(error).__name__,
                "message": f"Interrupted by {interruption_signal.name}",
                "phase": failed_phase,
                "signal": interruption_signal.name,
            }
            logger.warning("Training interrupted by {} during phase '{}'", interruption_signal.name, failed_phase)
        else:
            worklog["phase"] = "failed"
            worklog["status"] = "failed"
            worklog["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "phase": failed_phase,
            }
            logger.exception("Training failed during phase '{}': {}", failed_phase, error)
        raise
    finally:
        worklog["finished_at_utc"] = _utc_timestamp()
        worklog["elapsed_seconds"] = time.perf_counter() - run_started
        worklog["timings_seconds"] = timings
        worklog["resources"] = sampler.summary()
        _atomic_write_json(worklog_path, worklog)

    return worklog


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the TinyStories byte-level BPE tokenizer and write audit artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=_positive_int, default=10_000)
    parser.add_argument(
        "--max-documents",
        type=_positive_int,
        default=None,
        help=(
            "Copy only this many leading documents, ending at existing "
            "<|endoftext|> delimiters, into the training subset."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=_positive_int,
        default=DEFAULT_NUM_WORKERS,
        help="Maximum number of concurrent pre-tokenization worker processes.",
    )
    parser.add_argument(
        "--max-chunk-mib",
        type=_positive_int,
        default=DEFAULT_MAX_CHUNK_MIB,
        help=(
            "Target raw input size per pre-tokenization task. Chunks align to special-token boundaries, "
            "so a single long document can exceed this target."
        ),
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token to reserve; may be passed more than once.",
    )
    parser.add_argument(
        "--profile",
        dest="profile_mode",
        choices=("none", "cprofile"),
        default="none",
        help="cProfile records the parent process; use the worklog py-spy command for subprocesses.",
    )
    parser.add_argument(
        "--hash-input",
        action="store_true",
        help="Compute a SHA-256 fingerprint of the input before training.",
    )
    parser.add_argument(
        "--memory-sample-interval",
        type=_positive_float,
        default=0.25,
        help="RSS sampling interval in seconds for the process tree.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable tqdm progress bars while keeping log messages enabled.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    special_tokens = args.special_token if args.special_token is not None else DEFAULT_SPECIAL_TOKENS

    signal_handler = _ProcessTreeSignalHandler()
    try:
        with signal_handler:
            worklog = train_tinystories(
                input_path=args.input_path,
                output_dir=args.output_dir,
                vocab_size=args.vocab_size,
                special_tokens=special_tokens,
                profile_mode=args.profile_mode,
                hash_input=args.hash_input,
                memory_sample_interval=args.memory_sample_interval,
                max_documents=args.max_documents,
                show_progress=args.show_progress,
                num_workers=args.num_workers,
                max_chunk_mib=args.max_chunk_mib,
            )
    except KeyboardInterrupt:
        signal_handler.stop_descendants(force=True)
        logger.warning("Training interrupted by Ctrl-C")
        return 128 + signal.SIGINT
    except _SignalExit as error:
        signal_handler.stop_descendants(force=True)
        logger.warning("Training interrupted by {}", signal.Signals(error.signum).name)
        return 128 + error.signum
    except Exception as error:
        logger.error("Training failed: {}", error)
        return 1

    resources = worklog["resources"]
    analysis = worklog["analysis"]
    logger.success("Training complete")
    logger.info("Vocabulary: {}", worklog["artifacts"]["vocab"])
    logger.info("Merges: {}", worklog["artifacts"]["merges"])
    logger.info("Worklog: {}", worklog["artifacts"]["worklog"])
    logger.info("Run log: {}", worklog["artifacts"]["log"])
    logger.info("Training wall time: {:.3f}s", worklog["timings_seconds"]["training"])
    logger.info("Peak process-tree RSS: {:.3f} GiB", resources["peak_total_rss_gib"])
    if worklog["input"]["downsampling"]["enabled"]:
        logger.info("Training subset: {}", worklog["input"]["training_path"])
    logger.info("Longest token: {} bytes", analysis["longest_token"]["byte_length"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
