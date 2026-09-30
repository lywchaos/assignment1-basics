"""Tokenizer experiments from handout section 2.7 (PDF page 12, problem ``tokenizer_experiments``).

Measure the compression and throughput numbers with::

    uv run python -m cs336_basics.p12_tokenizer_experiments --steps sample throughput

Encode the four corpora into ``uint16`` token ID arrays for (d) with::

    uv run python -m cs336_basics.p12_tokenizer_experiments --steps encode-datasets

Results are printed and recorded in ``--output-dir/worklog.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from array import array
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from cs336_basics.p11_tokenizer import Tokenizer

DOCUMENT_DELIMITER = "<|endoftext|>"
SPECIAL_TOKENS = [DOCUMENT_DELIMITER]
DEFAULT_DATA_DIR = Path("data")
DEFAULT_TINYSTORIES_TOKENIZER_DIR = Path("artifacts/p9_tinystories")
DEFAULT_OWT_TOKENIZER_DIR = Path("artifacts/p10_openwebtext")
DEFAULT_OUTPUT_DIR = Path("artifacts/p12_tokenizer_experiments")
WORKLOG_FILENAME = "worklog.json"
PILE_BYTES = 825 * 1024**3  # handout section 2.7(c): "825GB of text"
ID_CHUNK_SIZE = 1 << 20
NPY_HEADER_LENGTH = 128  # multiple of 64, leaves room to patch the final shape
PROGRESS_INTERVAL_SECONDS = 30.0

TINYSTORIES = "tinystories"
OWT = "openwebtext"

# dataset name -> (file name, tokenizer key)
DATASETS: dict[str, tuple[str, str]] = {
    "tinystories_train": ("TinyStoriesV2-GPT4-train.txt", TINYSTORIES),
    "tinystories_valid": ("TinyStoriesV2-GPT4-valid.txt", TINYSTORIES),
    "owt_train": ("owt_train.txt", OWT),
    "owt_valid": ("owt_valid.txt", OWT),
}


def load_tokenizer(directory: Path) -> Tokenizer:
    return Tokenizer.from_files(str(directory / "vocab.json"), str(directory / "merges.txt"), SPECIAL_TOKENS)


def tokenizer_summary(tokenizer: Tokenizer) -> dict[str, object]:
    return {
        "vocab_size": len(tokenizer.vocab),
        "merge_count": len(tokenizer.merges),
        "special_tokens": tokenizer.special_tokens,
    }


def sample_documents(path: Path, num_documents: int, delimiter: str = DOCUMENT_DELIMITER) -> str:
    """Return the first ``num_documents`` documents, with their delimiter tokens included."""
    documents: list[str] = []
    with open(path, encoding="utf-8") as file:
        buffer = ""
        while len(documents) < num_documents:
            chunk = file.read(1 << 20)
            if not chunk:
                break
            buffer += chunk
            pieces = buffer.split(delimiter)
            buffer = pieces.pop()
            for piece in pieces:
                documents.append(piece + delimiter)
                if len(documents) == num_documents:
                    break
    return "".join(documents)


def read_text_budget(path: Path, max_bytes: int) -> str:
    """Read whole lines from ``path`` until ``max_bytes`` of UTF-8 text have been collected."""
    lines: list[str] = []
    total_bytes = 0
    with open(path, encoding="utf-8") as file:
        for line in file:
            lines.append(line)
            total_bytes += len(line.encode("utf-8"))
            if total_bytes >= max_bytes:
                break
    return "".join(lines)


def measure_compression(tokenizer: Tokenizer, text: str) -> dict[str, float | int]:
    source_bytes = len(text.encode("utf-8"))
    token_count = len(tokenizer.encode(text))
    return {
        "source_bytes": source_bytes,
        "token_count": token_count,
        "bytes_per_token": source_bytes / token_count if token_count else 0.0,
    }


def measure_throughput(tokenizer: Tokenizer, text: str) -> dict[str, float | int]:
    source_bytes = len(text.encode("utf-8"))
    started = time.perf_counter()
    token_count = sum(1 for _ in tokenizer.encode_iterable([text]))
    elapsed_seconds = time.perf_counter() - started
    bytes_per_second = source_bytes / elapsed_seconds
    return {
        "source_bytes": source_bytes,
        "token_count": token_count,
        "elapsed_seconds": elapsed_seconds,
        "bytes_per_second": bytes_per_second,
        "pile_hours": PILE_BYTES / bytes_per_second / 3600,
    }


def _npy_header(shape: tuple[int, ...]) -> bytes:
    header = f"{{'descr': '<u2', 'fortran_order': False, 'shape': {shape!r}, }}"
    padding = NPY_HEADER_LENGTH - 10 - len(header) - 1
    if padding < 0:
        raise ValueError(f"shape {shape!r} does not fit into the reserved npy header")
    return (header + " " * padding + "\n").encode("latin-1")


def _npy_prefix() -> bytes:
    return b"\x93NUMPY\x01\x00" + (NPY_HEADER_LENGTH - 10).to_bytes(2, "little")


def _id_chunks(token_ids: Iterator[int], chunk_size: int = ID_CHUNK_SIZE) -> Iterator[array]:
    chunk = array("H")
    for token_id in token_ids:
        chunk.append(token_id)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = array("H")
    if chunk:
        yield chunk


def _format_bytes(num_bytes: float) -> str:
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    value = num_bytes
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024
        if value < 1024:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PiB"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def encode_dataset(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    *,
    label: str | None = None,
    progress_interval: float = PROGRESS_INTERVAL_SECONDS,
) -> int:
    """Stream ``input_path`` into a ``uint16`` ``.npy`` array without loading the corpus into memory."""
    label = label or input_path.name
    total_bytes = input_path.stat().st_size
    started = time.perf_counter()
    last_report = started
    bytes_read = 0
    total_tokens = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"(d) {label}: encoding {_format_bytes(total_bytes)} -> {output_path}", flush=True)

    with open(output_path, "wb") as output, open(input_path, encoding="utf-8") as source:

        def counting_lines() -> Iterator[str]:
            nonlocal bytes_read
            for line in source:
                bytes_read += len(line.encode("utf-8"))
                yield line

        def report(force: bool = False) -> None:
            nonlocal last_report
            now = time.perf_counter()
            if not force and now - last_report < progress_interval:
                return
            elapsed = now - started
            rate = bytes_read / elapsed if elapsed else 0.0
            percent = 100 * bytes_read / total_bytes if total_bytes else 100.0
            eta = max(total_bytes - bytes_read, 0) / rate if rate else 0.0
            print(
                f"(d) {label}: {percent:5.1f}% ({_format_bytes(bytes_read)}/{_format_bytes(total_bytes)}), "
                f"{total_tokens:,} tokens, {_format_bytes(rate)}/s, ETA {_format_duration(eta)}",
                flush=True,
            )
            last_report = now

        # Provenance: the .npy files under artifacts/p12_tokenizer_experiments/ were generated by
        # the streaming encoder at commit ca7af3f, i.e. before the readability refactor of
        # Tokenizer.encode_iterable (the recovery command lives in that method's comment). Since
        # the two versions were verified per-ID equivalent, these ids stay valid unless that
        # equivalence is ever disproven -- in that case rerun the encode-datasets step instead of
        # patching the files.
        output.write(_npy_prefix())
        output.write(_npy_header((0,)))
        for chunk in _id_chunks(tokenizer.encode_iterable(counting_lines())):
            output.write(np.asarray(chunk, dtype=np.uint16).astype("<u2", copy=False).tobytes())
            total_tokens += len(chunk)
            if len(chunk) == ID_CHUNK_SIZE:
                report()
        output.seek(0)
        output.write(_npy_prefix())
        output.write(_npy_header((total_tokens,)))
        report(force=True)
    return total_tokens


def run_sample(args: argparse.Namespace, tokenizers: dict[str, Tokenizer]) -> dict[str, dict[str, float | int]]:
    """(a) and (b): compression ratios on 10 sampled documents per corpus."""
    print(f"(a)/(b) sampling {args.num_sample_docs} documents per corpus", flush=True)
    tinystories_text = sample_documents(args.data_dir / DATASETS["tinystories_train"][0], args.num_sample_docs)
    owt_text = sample_documents(args.data_dir / DATASETS["owt_train"][0], args.num_sample_docs)

    reports = {
        "tinystories_train / tinystories tokenizer": measure_compression(tokenizers[TINYSTORIES], tinystories_text),
        "owt_train / openwebtext tokenizer": measure_compression(tokenizers[OWT], owt_text),
        "owt_train / tinystories tokenizer": measure_compression(tokenizers[TINYSTORIES], owt_text),
    }
    for name, report in reports.items():
        print(
            f"(a)/(b) {name}: {report['bytes_per_token']:.3f} bytes/token "
            f"({report['token_count']:,} tokens from {report['source_bytes']:,} bytes)",
            flush=True,
        )
    return reports


def run_throughput(args: argparse.Namespace, tokenizers: dict[str, Tokenizer]) -> dict[str, dict[str, float | int]]:
    """(c): bytes/second on a bounded text sample and the projected Pile time."""
    reports: dict[str, dict[str, float | int]] = {}
    print(f"(c) measuring throughput on ~{args.throughput_mib} MiB per corpus", flush=True)
    for dataset_name in ("tinystories_train", "owt_train"):
        file_name, tokenizer_key = DATASETS[dataset_name]
        text = read_text_budget(args.data_dir / file_name, args.throughput_mib * 2**20)
        report = measure_throughput(tokenizers[tokenizer_key], text)
        reports[f"{dataset_name} / {tokenizer_key} tokenizer"] = report
        print(
            f"(c) {dataset_name}: {report['token_count']:,} tokens from {report['source_bytes']:,} bytes "
            f"in {report['elapsed_seconds']:.2f}s -> {report['bytes_per_second'] / 2**20:.2f} MiB/s, "
            f"Pile (825GB) -> {report['pile_hours']:.0f} hours",
            flush=True,
        )
    return reports


def run_encode_datasets(
    args: argparse.Namespace, tokenizers: dict[str, Tokenizer]
) -> dict[str, dict[str, str | float | int]]:
    """(d): encode each requested corpus into a uint16 ``.npy`` token ID array."""
    reports: dict[str, dict[str, str | float | int]] = {}
    for dataset_name in args.datasets:
        file_name, tokenizer_key = DATASETS[dataset_name]
        input_path = args.data_dir / file_name
        output_path = args.output_dir / f"{dataset_name}_ids.npy"
        started = time.perf_counter()
        token_count = encode_dataset(tokenizers[tokenizer_key], input_path, output_path, label=dataset_name)
        elapsed_seconds = time.perf_counter() - started
        reports[dataset_name] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "token_count": token_count,
            "elapsed_seconds": elapsed_seconds,
            "bytes_per_second": input_path.stat().st_size / elapsed_seconds,
        }
        print(
            f"(d) {dataset_name}: done, {token_count:,} tokens in {_format_duration(elapsed_seconds)} "
            f"({_format_bytes(input_path.stat().st_size / elapsed_seconds)}/s) -> {output_path}",
            flush=True,
        )
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tinystories-tokenizer-dir", type=Path, default=DEFAULT_TINYSTORIES_TOKENIZER_DIR)
    parser.add_argument("--owt-tokenizer-dir", type=Path, default=DEFAULT_OWT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=("sample", "throughput", "encode-datasets"),
        default=["sample", "throughput"],
        help="(d) takes hours on the full corpora, so it is opt-in.",
    )
    parser.add_argument("--num-sample-docs", type=int, default=10)
    parser.add_argument("--throughput-mib", type=int, default=8)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"p12 tokenizer experiments: steps={', '.join(args.steps)}", flush=True)

    tokenizers = {
        TINYSTORIES: load_tokenizer(args.tinystories_tokenizer_dir),
        OWT: load_tokenizer(args.owt_tokenizer_dir),
    }
    worklog: dict[str, object] = {
        "steps": args.steps,
        "tokenizers": {name: tokenizer_summary(tokenizer) for name, tokenizer in tokenizers.items()},
    }
    if "sample" in args.steps:
        worklog["sample"] = run_sample(args, tokenizers)
    if "throughput" in args.steps:
        worklog["throughput"] = run_throughput(args, tokenizers)
    if "encode-datasets" in args.steps:
        worklog["encode-datasets"] = run_encode_datasets(args, tokenizers)

    worklog_path = args.output_dir / WORKLOG_FILENAME
    worklog_path.write_text(json.dumps(worklog, indent=2) + "\n", encoding="utf-8")
    print(f"worklog written to {worklog_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
