"""Same-graph paraphrase detection with hard topology-breaking negatives."""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset

from .semantic_story import StoryGraphDataset


@dataclass(frozen=True)
class StructuralParaphraseExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    same_graph: bool


def render_pair_example(
    *,
    seed: int,
    graph_id: str,
    left: str,
    right: str,
    same_graph: bool,
) -> StructuralParaphraseExample:
    prefix = f"Story one: {left} Story two: {right} Do these stories have the same event graph? Answer"
    answer = "yes" if same_graph else "no"
    return StructuralParaphraseExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        same_graph=same_graph,
    )


class StructuralParaphraseDataset(Dataset):
    """Balanced same-graph vs hard-negative story pairs."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        variants_per_graph: int = 4,
        hard_negatives_per_graph: int = 3,
    ) -> None:
        n_graphs = max(1, (n_examples + 1) // 2)
        base = StoryGraphDataset(
            n_graphs=n_graphs,
            variants_per_graph=variants_per_graph,
            hard_negatives_per_graph=hard_negatives_per_graph,
            seed=seed,
        )
        rng = random.Random(seed)
        examples: list[StructuralParaphraseExample] = []
        for idx in range(n_examples):
            row = base.rows[idx % len(base.rows)]
            positives = list(row["positives"])  # type: ignore[arg-type]
            negatives = list(row["hard_negatives"])  # type: ignore[arg-type]
            same_graph = idx % 2 == 0
            left = rng.choice(positives)
            if same_graph:
                right_choices = [text for text in positives if text != left]
                right = rng.choice(right_choices if right_choices else positives)
            else:
                right = rng.choice(negatives)
            graph = row["graph"]
            graph_id = getattr(graph, "graph_id", f"sp{idx}")
            examples.append(
                render_pair_example(
                    seed=seed + idx,
                    graph_id=graph_id,
                    left=left,
                    right=right,
                    same_graph=same_graph,
                )
            )
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> StructuralParaphraseExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[StructuralParaphraseExample]) -> list[StructuralParaphraseExample]:
    return batch
