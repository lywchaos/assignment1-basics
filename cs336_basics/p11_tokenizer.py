"""Byte-level BPE tokenizer: encode text into token IDs and decode IDs back into text.

Covers handout section 2.6 (PDF page 11, problem ``tokenizer``, 15 points).

Contract from the handout:

* ``encode`` pre-tokenizes the input with the same regex used during training, then applies
  the learned merges in creation order, independently within each pre-token (no merges may
  cross pre-token boundaries).
* Special tokens are never split: they are matched before pre-tokenization and each maps to
  a single ID. A special token missing from ``vocab`` has to be appended.
* ``decode`` concatenates the bytes of the requested IDs and decodes them as UTF-8 with
  ``errors="replace"``, since an arbitrary ID sequence need not be valid UTF-8.
* ``encode_iterable`` has to stay correct under chunked input: the tests compare it against
  ``encode`` on the concatenated text and cap the function's memory at 1 MB.

Wire ``tests/adapters.py::get_tokenizer`` to this class, then check your work with::

    uv run pytest tests/test_tokenizer.py
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator

from cs336_basics.p9_bpe_tokenizer_training import PAT

# Encoding has to pre-tokenize exactly like training, otherwise the learned merges get
# applied to words that never existed during training. Reuse that regex instead of copying it.
PRETOKEN_PATTERN = PAT

Merge = tuple[bytes, bytes]


def gpt2_unicode_to_bytes() -> dict[str, int]:
    """Return the printable-character -> byte map needed to load serialized artifacts.

    ``cs336_basics/p9_train_bpe_tinystories.py`` serializes ``vocab.json`` and ``merges.txt``
    with GPT-2's reversible byte-to-unicode encoding (``_gpt2_bytes_to_unicode``); loading
    needs the inverse map. Either invert that function here or promote the mapping into a
    shared module so serialization and loading cannot drift apart.
    """
    raise NotImplementedError("TODO(p11): build the inverse GPT-2 byte map")


class Tokenizer:
    """BPE tokenizer built from a vocabulary and an ordered list of merges."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Merge],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Build a tokenizer from ``vocab``, ``merges`` and optional ``special_tokens``.

        ``vocab`` maps token IDs to raw bytes; ``merges`` is ordered by creation time, which
        is also the order in which an encoded pre-token has to consume them.

        TODO(p11): store the inputs and precompute the derived state the other methods need
        (token -> ID lookup, merge ranking, special tokens to match first). ``special_tokens``
        must include tokens that are missing from ``vocab``: append them instead of mutating
        the caller's dictionary.
        """
        raise NotImplementedError("TODO(p11): build the tokenizer state")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load a tokenizer from the artifacts written by the BPE training code.

        TODO(p11): read ``vocab.json`` and ``merges.txt`` in the GPT-2 printable format
        produced by ``cs336_basics/p9_train_bpe_tinystories.py::serialize_vocab`` and
        ``serialize_merges``, convert both back to bytes, then call the constructor.
        """
        raise NotImplementedError("TODO(p11): load vocab.json and merges.txt")

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token IDs.

        TODO(p11): split ``text`` on special tokens first (a special token always becomes one
        ID), pre-tokenize the remaining pieces, then apply the merge list to each pre-token.
        """
        raise NotImplementedError("TODO(p11): encode text")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily yield token IDs for an iterable of strings, e.g. a file handle.

        The memory footprint must stay constant: the caller may hand over a 5 MB file line by
        line while the function runs under a 1 MB ``RLIMIT_AS``. Chunked input must still
        tokenize exactly like a single ``encode`` over the concatenated text, so the chunks
        cannot be treated as independent documents.

        TODO(p11): make this a generator that consumes ``iterable`` incrementally.
        """
        raise NotImplementedError("TODO(p11): stream encode an iterable")

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs back into text.

        TODO(p11): concatenate the vocabulary bytes for ``ids`` and decode UTF-8 with
        ``errors="replace"``.
        """
        raise NotImplementedError("TODO(p11): decode token IDs")
