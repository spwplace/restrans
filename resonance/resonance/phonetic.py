"""Phonetic utilities for rhyme-based phase initialisation.

Uses the `pronouncing` library (a CMU Pronouncing Dictionary wrapper) to
map tokens to rhyme groups, enabling structural phase embeddings.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

try:
    import pronouncing
except ImportError as _err:  # pragma: no cover
    pronouncing = None  # type: ignore[assignment]


def build_rhyme_index(tokenizer, vocab_size: int) -> dict[str, Any]:
    """Build a hierarchical rhyme-based index for the tokenizer vocabulary.

    Iterates over every token ID, decodes it to a string, cleans it to
    alphabetic characters, and queries the CMU dictionary for phonemes.
    Tokens that share a rhyme part (the stressed vowel and everything
    after it) are grouped together.

    Args:
        tokenizer: A Hugging Face ``PreTrainedTokenizer``-like object with
            ``decode(token_ids: list[int]) -> str``.
        vocab_size: Size of the tokenizer vocabulary to iterate over.

    Returns:
        A dictionary with the following keys:
        - ``token_to_rhyme``: ``dict[int, str]`` mapping token ID to rhyme.
        - ``rhyme_to_tokens``: ``dict[str, list[int]]`` mapping rhyme to token IDs.
        - ``token_to_phones``: ``dict[int, str]`` mapping token ID to full phonemes.
        - ``n_rhyme_groups``: Number of unique rhyme groups found.
        - ``tokens_with_rhyme``: Number of tokens that mapped to a rhyme.

    Raises:
        ImportError: If the ``pronouncing`` package is not installed.
    """
    if pronouncing is None:
        raise ImportError(
            "The 'pronouncing' package is required for phonetic initialisation. "
            "Install it with: pip install pronouncing"
        )

    token_to_rhyme: dict[int, str] = {}
    rhyme_to_tokens: defaultdict[str, list[int]] = defaultdict(list)
    token_to_phones: dict[int, str] = {}

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id]).strip().lower()
        token_clean = "".join(c for c in token_str if c.isalpha())

        if len(token_clean) < 2:
            continue

        phones = pronouncing.phones_for_word(token_clean)
        if phones:
            rhyme_part = pronouncing.rhyming_part(phones[0])
            token_to_rhyme[token_id] = rhyme_part
            rhyme_to_tokens[rhyme_part].append(token_id)
            token_to_phones[token_id] = phones[0]

    rhyme_index = {
        "token_to_rhyme": token_to_rhyme,
        "rhyme_to_tokens": dict(rhyme_to_tokens),
        "token_to_phones": token_to_phones,
        "n_rhyme_groups": len(rhyme_to_tokens),
        "tokens_with_rhyme": len(token_to_rhyme),
    }
    return rhyme_index
