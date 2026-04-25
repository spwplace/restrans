"""Dataset classes for causal language modelling."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Fixed-length tokenized chunks for causal language modelling.

    Given a sequence of text documents, each document is tokenised and split
    into non-overlapping chunks of ``max_length`` tokens.  No padding is
    required because every example has exactly ``max_length`` tokens.

    Each example yields ``{"input_ids": ids, "labels": ids}`` so that the
    training loop can compute next-token prediction loss.
    """

    def __init__(
        self,
        documents,
        tokenizer,
        max_length: int = 256,
        max_examples: int = 200_000,
    ) -> None:
        """Create a ``TextDataset``.

        Args:
            documents: An iterable of dict-like objects with a ``"text"`` key.
            tokenizer: A tokenizer with ``encode(text, truncation=False)``.
            max_length: Number of tokens per fixed-length chunk.
            max_examples: Upper bound on the number of chunks to retain.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[list[int]] = []

        for i, doc in enumerate(documents):
            tokens = tokenizer.encode(doc["text"], truncation=False)

            # Create non-overlapping chunks
            for j in range(0, len(tokens) - max_length + 1, max_length):
                self.examples.append(tokens[j : j + max_length])
                if len(self.examples) >= max_examples:
                    break

            if len(self.examples) >= max_examples:
                break

            if (i + 1) % 20_000 == 0:
                print(f"  Processed {i + 1} documents, {len(self.examples)} examples")

        if len(self.examples) == 0:
            raise ValueError(
                "TextDataset produced 0 examples. "
                f"Documents may all be shorter than max_length={max_length}. "
                "Consider using longer documents or reducing max_length."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens}
