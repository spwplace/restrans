"""Dyck and cross-serial bracket validation tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset


PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
OPEN_TO_CLOSE = dict(PAIRS)
CLOSE_TO_OPEN = {close: open_ for open_, close in PAIRS}


@dataclass(frozen=True)
class DyckExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    valid: bool
    mode: str
    depth: int


def _nested_sequence(rng: random.Random, depth: int, n_types: int) -> list[str]:
    opens = [rng.choice(PAIRS[:n_types])[0] for _ in range(depth)]
    closes = [OPEN_TO_CLOSE[token] for token in reversed(opens)]
    return opens + closes


def _cross_serial_sequence(rng: random.Random, depth: int, n_types: int) -> list[str]:
    opens = [rng.choice(PAIRS[:n_types])[0] for _ in range(depth)]
    closes = [OPEN_TO_CLOSE[token] for token in opens]
    return opens + closes


def _corrupt(seq: list[str], rng: random.Random, n_types: int) -> list[str]:
    out = list(seq)
    close_positions = [idx for idx, token in enumerate(out) if token in CLOSE_TO_OPEN]
    idx = rng.choice(close_positions if close_positions else list(range(len(out))))
    choices = [close for _, close in PAIRS[:n_types] if close != out[idx]]
    out[idx] = rng.choice(choices)
    return out


def render_dyck_example(
    *,
    seed: int,
    graph_id: str,
    max_depth: int = 8,
    n_types: int = 3,
    mode: str = "nested",
    positive: bool = True,
) -> DyckExample:
    rng = random.Random(seed)
    depth = rng.randint(1, max_depth)
    if mode == "cross":
        seq = _cross_serial_sequence(rng, depth, n_types)
        rule = "opens must close in the same order"
    elif mode == "nested":
        seq = _nested_sequence(rng, depth, n_types)
        rule = "opens must close in reverse order"
    else:
        raise ValueError(f"unknown dyck mode: {mode}")
    if not positive:
        seq = _corrupt(seq, rng, n_types)
    text = " ".join(seq)
    prefix = f"Rule: {rule}. Sequence: {text}. Is the sequence well formed? Answer"
    answer = "yes" if positive else "no"
    return DyckExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        valid=positive,
        mode=mode,
        depth=depth,
    )


class DyckDataset(Dataset):
    """Balanced yes/no bracket-language validation examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        max_depth: int = 8,
        n_types: int = 3,
        mode: str = "nested",
    ) -> None:
        self.examples = [
            render_dyck_example(
                seed=seed + idx,
                graph_id=f"dyck{idx}",
                max_depth=max_depth,
                n_types=n_types,
                mode=mode,
                positive=idx % 2 == 0,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DyckExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[DyckExample]) -> list[DyckExample]:
    return batch
