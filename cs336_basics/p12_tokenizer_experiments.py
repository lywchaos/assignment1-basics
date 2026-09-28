"""Tokenizer experiments from handout section 2.7 (PDF page 12, problem ``tokenizer_experiments``).

Steps, matching the four deliverables:

* (a) Sample 10 documents per corpus and report the compression ratio (bytes/token) for the
  matching tokenizer (TinyStories 10K, OpenWebText 32K).
* (b) Encode the OpenWebText sample with the TinyStories tokenizer and compare.
* (c) Measure throughput in bytes/second and extrapolate to the Pile (825 GB of text).
* (d) Encode both training sets and both development sets into ``uint16`` token ID arrays;
  the vocabularies are smaller than ``2**16``, so IDs fit without overflow.

Run it with::

    uv run python -m cs336_basics.p12_tokenizer_experiments \
        --output-dir artifacts/p12_tokenizer_experiments

A run writes ``worklog.json`` (audit record of the measurements), ``tokenizer_experiments.log``
and the token ID arrays from (d) into ``--output-dir``.
"""

from __future__ import annotations

import argparse
import signal
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from cs336_basics.p11_tokenizer import Tokenizer

# Reuse the p9 runner's experiment plumbing instead of keeping a second copy.
# TODO(p12): if more shared helpers appear, promote them into a module such as
# ``cs336_basics/experiment_io.py`` and update both runners.
from cs336_basics.p9_train_bpe_tinystories import _atomic_write_json, _configure_logging, _positive_int

DEFAULT_DATA_DIR = Path("data")
DEFAULT_TINYSTORIES_TOKENIZER_DIR = Path("artifacts/p9_tinystories")
DEFAULT_OWT_TOKENIZER_DIR = Path("artifacts/p10_openwebtext")
DEFAULT_OUTPUT_DIR = Path("artifacts/p12_tokenizer_experiments")
DEFAULT_SPECIAL_TOKENS = ("<|endoftext|>",)
DOCUMENT_DELIMITER = "<|endoftext|>"
VOCAB_FILENAME = "vocab.json"
MERGES_FILENAME = "merges.txt"
WORKLOG_FILENAME = "worklog.json"
LOG_FILENAME = "tokenizer_experiments.log"
PILE_BYTES = 825 * 1024**3  # "825 GB of text", handout section 2.7(c)
STEP_CHOICES = ("sample", "throughput", "encode-datasets")

# Source files needed for (d), as ``<corpus>_<split>``; resolve them against --data-dir.
DATASET_FILENAMES = {
    "tinystories_train": "TinyStoriesV2-GPT4-train.txt",
    "tinystories_valid": "TinyStoriesV2-GPT4-valid.txt",
    "owt_train": "owt_train.txt",
    "owt_valid": "owt_valid.txt",
}


@dataclass(frozen=True)
class CompressionReport:
    """Compression measured on one dataset with one tokenizer (deliverables (a) and (b))."""

    dataset: str
    tokenizer: str
    documents: int
    source_bytes: int
    token_count: int


@dataclass(frozen=True)
class ThroughputReport:
    """Timing of one encode pass; the derived fields keep the worklog self-contained."""

    source_bytes: int
    seconds: float
    bytes_per_second: float
    pile_hours: float


def load_tokenizer(directory: Path, special_tokens: Sequence[str]) -> Tokenizer:
    """Load a trained tokenizer from a p9/p10 artifact directory.

    TODO(p12): the artifacts are ``directory / VOCAB_FILENAME`` and
    ``directory / MERGES_FILENAME``, both loadable through ``Tokenizer.from_files``.
    """
    raise NotImplementedError("TODO(p12): load the tokenizer artifacts")


def sample_documents(path: Path, num_documents: int, delimiter: str = DOCUMENT_DELIMITER) -> list[str]:
    """(a) Return the first ``num_documents`` documents of ``path``.

    TODO(p12): stream the file instead of reading all of it; TinyStories is ~2 GB and
    OpenWebText ~11 GB. Documents end at ``delimiter``, and keeping the delimiter in the
    sampled text is what makes the byte counts comparable across corpora.
    """
    raise NotImplementedError("TODO(p12): sample documents")


def measure_compression(
    tokenizer: Tokenizer,
    documents: Sequence[str],
    *,
    dataset: str,
    tokenizer_name: str,
) -> CompressionReport:
    """(a)/(b) Encode ``documents`` and return the byte/token accounting.

    TODO(p12): count source bytes and token IDs; the reported ratio is
    ``source_bytes / token_count``.
    """
    raise NotImplementedError("TODO(p12): measure compression")


def measure_throughput(tokenizer: Tokenizer, text: str) -> ThroughputReport:
    """(c) Time one encode pass over ``text``.

    TODO(p12): measure wall-clock time around ``tokenizer.encode``, then derive
    bytes/second and the hours needed for the Pile (``PILE_BYTES / bytes_per_second / 3600``).
    """
    raise NotImplementedError("TODO(p12): measure throughput")


def encode_dataset(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> int:
    """(d) Encode one corpus split into a ``uint16`` token ID array.

    TODO(p12): stream ``input_path`` through ``tokenizer.encode_iterable`` (one file handle,
    not the whole file) and write the IDs to ``output_path`` without materializing the full
    Python list; return the number of IDs written.
    """
    raise NotImplementedError("TODO(p12): encode a dataset")


def run_experiments(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested steps and return the worklog payload.

    TODO(p12): load both tokenizers, run (a)-(b) for ``sample``, (c) for ``throughput`` and
    (d) for ``encode-datasets``, then record inputs, parameters, reports and artifact paths
    as JSON-serializable data.
    """
    raise NotImplementedError("TODO(p12): run the tokenizer experiments")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the handout's tokenizer experiments and write audit artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory holding the raw corpora.")
    parser.add_argument(
        "--tinystories-tokenizer-dir",
        type=Path,
        default=DEFAULT_TINYSTORIES_TOKENIZER_DIR,
        help="Artifact directory of the 10K TinyStories tokenizer.",
    )
    parser.add_argument(
        "--owt-tokenizer-dir",
        type=Path,
        default=DEFAULT_OWT_TOKENIZER_DIR,
        help="Artifact directory of the 32K OpenWebText tokenizer.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_CHOICES,
        default=list(STEP_CHOICES),
        help="Which experiments to run; (d) is the expensive one.",
    )
    parser.add_argument(
        "--num-sample-docs",
        type=_positive_int,
        default=10,
        help="Documents per corpus for the compression measurements in (a) and (b).",
    )
    parser.add_argument(
        "--throughput-mib",
        type=_positive_int,
        default=16,
        help="Raw input size encoded by the throughput measurement in (c).",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token to pass to the tokenizer; may be passed more than once.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable tqdm progress bars while keeping log messages enabled.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(output_dir / LOG_FILENAME, args.show_progress)

    try:
        worklog = run_experiments(args)
    except KeyboardInterrupt:
        logger.warning("Tokenizer experiments interrupted by Ctrl-C")
        return 128 + signal.SIGINT
    except Exception as error:
        logger.error("Tokenizer experiments failed: {}", error)
        return 1

    worklog_path = output_dir / WORKLOG_FILENAME
    _atomic_write_json(worklog_path, worklog)
    logger.success("Tokenizer experiments complete")
    logger.info("Worklog: {}", worklog_path)
    logger.info("Run log: {}", output_dir / LOG_FILENAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
