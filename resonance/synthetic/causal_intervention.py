"""Controlled causal-intervention questions over rendered event graphs."""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset


EVENTS = [
    "the alarm rang",
    "the gate opened",
    "the lamp flashed",
    "the cart moved",
    "the bell chimed",
    "the door locked",
    "the pump started",
    "the flag rose",
]


@dataclass(frozen=True)
class CausalGraph:
    events: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]

    def parents(self, node: int) -> set[int]:
        return {source for source, target in self.edges if target == node}


@dataclass(frozen=True)
class CausalInterventionExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    intervention: int
    target: int
    target_happens: bool


def generate_causal_graph(
    *,
    seed: int,
    n_events: int = 6,
    edge_prob: float = 0.35,
) -> CausalGraph:
    rng = random.Random(seed)
    event_texts = tuple(rng.sample(EVENTS, n_events))
    edges: set[tuple[int, int]] = set()
    for idx in range(n_events - 1):
        edges.add((idx, idx + 1))
    for source in range(n_events):
        for target in range(source + 2, n_events):
            if rng.random() < edge_prob:
                edges.add((source, target))
    return CausalGraph(events=event_texts, edges=tuple(sorted(edges)))


def active_events(graph: CausalGraph, removed: int | None = None) -> set[int]:
    active: set[int] = set()
    for node in range(len(graph.events)):
        if node == removed:
            continue
        parents = graph.parents(node)
        if not parents:
            active.add(node)
            continue
        if any(parent in active for parent in parents):
            active.add(node)
    return active


def render_causal_example(
    *,
    seed: int,
    graph_id: str,
    n_events: int = 6,
    edge_prob: float = 0.35,
    positive: bool = True,
) -> CausalInterventionExample:
    rng = random.Random(seed)
    for attempt in range(1_000):
        graph = generate_causal_graph(seed=seed + attempt * 997, n_events=n_events, edge_prob=edge_prob)
        baseline = active_events(graph)
        candidates: list[tuple[int, int, bool]] = []
        for intervention in range(n_events):
            after = active_events(graph, removed=intervention)
            for target in baseline:
                if target == intervention:
                    continue
                target_happens = target in after
                if target_happens == positive:
                    candidates.append((intervention, target, target_happens))
        if candidates:
            intervention, target, target_happens = rng.choice(candidates)
            break
    else:
        raise RuntimeError("failed to generate causal intervention example")

    event_lines = [f"Event {chr(ord('A') + idx)}: {text}." for idx, text in enumerate(graph.events)]
    edge_lines = [
        f"Event {chr(ord('A') + source)} caused event {chr(ord('A') + target)}."
        for source, target in graph.edges
    ]
    rng.shuffle(edge_lines)
    prefix = (
        " ".join(event_lines + edge_lines)
        + f" If event {chr(ord('A') + intervention)} had not happened, "
        + f"would event {chr(ord('A') + target)} still happen? Answer"
    )
    answer = "yes" if target_happens else "no"
    return CausalInterventionExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        intervention=intervention,
        target=target,
        target_happens=target_happens,
    )


class CausalInterventionDataset(Dataset):
    """Balanced yes/no counterfactual causal graph questions."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        n_events: int = 6,
        edge_prob: float = 0.35,
    ) -> None:
        self.examples = [
            render_causal_example(
                seed=seed + idx,
                graph_id=f"causal{idx}",
                n_events=n_events,
                edge_prob=edge_prob,
                positive=idx % 2 == 0,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> CausalInterventionExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[CausalInterventionExample]) -> list[CausalInterventionExample]:
    return batch
