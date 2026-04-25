"""Shared utilities for the Resonance transformer package."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility across NumPy, Python, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_module_params(module: torch.nn.Module) -> int:
    """Return the total number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def generate_synthetic_text(
    n_documents: int = 10_000,
    min_words: int = 50,
    max_words: int = 500,
    seed: int | None = 42,
) -> list[dict[str, str]]:
    """Generate synthetic documents for quick testing.

    The generated text is structured but nonsensical English-like prose,
    assembled from a small vocabulary so that the tokenizer does not need
    to handle out-of-vocabulary issues.

    Args:
        n_documents: Number of documents to generate.
        min_words: Minimum words per document.
        max_words: Maximum words per document.
        seed: Optional random seed.

    Returns:
        List of dicts with a ``"text"`` key containing the document string.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    vocab = [
        "the", "a", "an", "in", "on", "at", "to", "from", "with", "by",
        "cat", "dog", "bird", "fish", "tree", "river", "mountain", "sky",
        "runs", "jumps", "swims", "flies", "sings", "dances", "dreams",
        "happily", "quickly", "slowly", "brightly", "darkly", "softly",
        "big", "small", "red", "blue", "green", "old", "new", "young",
        "and", "but", "or", "so", "yet", "because", "although", "while",
        "sun", "moon", "star", "cloud", "rain", "wind", "snow", "fire",
        "time", "space", "world", "life", "story", "song", "dream", "path",
        "Alice", "Bob", "Eve", "Mallory", "Trent", "Wendy", "Oscar",
        "loves", "hates", "finds", "loses", "creates", "destroys", "sees",
        "morning", "evening", "night", "day", "dawn", "dusk", "noon",
        "went", "came", "arrived", "departed", "returned", "left",
        "house", "castle", "garden", "forest", "ocean", "desert", "city",
        "laughter", "silence", "music", "noise", "peace", "chaos",
    ]

    documents: list[dict[str, str]] = []
    for _ in range(n_documents):
        n_words = rng.randint(min_words, max_words)
        words = [rng.choice(vocab) for _ in range(n_words)]
        text = " ".join(words)
        text = text[0].upper() + text[1:] + "."
        documents.append({"text": text})

    return documents
