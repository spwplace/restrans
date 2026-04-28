"""Aliased graph-walk question generator.

Each graph has latent node IDs, but every node is rendered with several
surface aliases.  The prompt gives alias equivalence classes and edges using
one alias per node.  Walk prefixes and candidate next nodes use different
aliases, so lexical matching is not enough: the model must canonicalize aliases
and use graph adjacency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset


SYLLABLES = [
    "ba", "ce", "di", "fo", "ga", "hi", "jo", "ka", "lu", "mi",
    "no", "pa", "qu", "ra", "su", "ti", "ve", "wo", "xe", "ya", "zo",
]


@dataclass(frozen=True)
class GraphAliasExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    current_node: int
    candidate_node: int


@dataclass(frozen=True)
class AliasGraph:
    graph_id: str
    aliases: tuple[tuple[str, ...], ...]
    edges: tuple[tuple[int, int], ...]

    def successors(self, node: int) -> set[int]:
        return {target for source, target in self.edges if source == node}


def _make_word(rng: random.Random, used: set[str]) -> str:
    for _ in range(10_000):
        word = "".join(rng.choice(SYLLABLES) for _ in range(3))
        if word not in used:
            used.add(word)
            return word
    raise RuntimeError("failed to generate unique alias")


def make_alias_pool(*, seed: int, size: int) -> tuple[str, ...]:
    rng = random.Random(seed)
    used: set[str] = set()
    return tuple(_make_word(rng, used) for _ in range(size))


def generate_alias_graph(
    *,
    seed: int,
    graph_id: str,
    edge_seed: int | None = None,
    alias_pool: tuple[str, ...] | None = None,
    n_nodes: int = 16,
    aliases_per_node: int = 3,
    out_degree: int = 2,
) -> AliasGraph:
    rng = random.Random(seed)
    alias_count = n_nodes * aliases_per_node
    if alias_pool is None:
        used: set[str] = set()
        flat_aliases = [_make_word(rng, used) for _ in range(alias_count)]
    else:
        if len(alias_pool) < alias_count:
            raise ValueError("alias_pool is too small for graph shape")
        flat_aliases = rng.sample(list(alias_pool), alias_count)
    aliases = tuple(
        tuple(flat_aliases[start : start + aliases_per_node])
        for start in range(0, alias_count, aliases_per_node)
    )
    edge_rng = rng if edge_seed is None else random.Random(edge_seed)
    edges: set[tuple[int, int]] = set()
    for source in range(n_nodes):
        candidates = [node for node in range(n_nodes) if node != source]
        for target in edge_rng.sample(candidates, min(out_degree, len(candidates))):
            edges.add((source, target))
    return AliasGraph(graph_id=graph_id, aliases=aliases, edges=tuple(sorted(edges)))


def _alias(graph: AliasGraph, node: int, variant: int) -> str:
    aliases = graph.aliases[node]
    return aliases[variant % len(aliases)]


def render_alias_facts(graph: AliasGraph) -> list[str]:
    lines = []
    for aliases in graph.aliases:
        if len(aliases) == 2:
            lines.append(f"{aliases[0]} and {aliases[1]} name the same place.")
        else:
            head = ", ".join(aliases[:-1])
            lines.append(f"{head}, and {aliases[-1]} name the same place.")
    return lines


def render_edge_facts(graph: AliasGraph) -> list[str]:
    return [
        f"{_alias(graph, source, 0)} leads to {_alias(graph, target, 0)}."
        for source, target in graph.edges
    ]


def sample_walk(
    graph: AliasGraph,
    rng: random.Random,
    *,
    length: int,
) -> list[int]:
    node = rng.randrange(len(graph.aliases))
    walk = [node]
    for _ in range(length - 1):
        successors = sorted(graph.successors(node))
        if not successors:
            break
        node = rng.choice(successors)
        walk.append(node)
    return walk


def exact_reachable(graph: AliasGraph, source: int, steps: int) -> set[int]:
    frontier = {source}
    for _ in range(steps):
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(graph.successors(node))
        frontier = next_frontier
    return frontier


def render_example(
    graph: AliasGraph,
    *,
    seed: int,
    walk_length: int = 4,
    query_steps: int = 1,
    positive: bool = True,
) -> GraphAliasExample:
    rng = random.Random(seed)
    walk = sample_walk(graph, rng, length=walk_length)
    current = walk[-1]
    reachable = sorted(exact_reachable(graph, current, query_steps))
    non_successors = [
        node
        for node in range(len(graph.aliases))
        if node not in reachable
    ]
    if positive and reachable:
        candidate = rng.choice(reachable)
    elif non_successors:
        candidate = rng.choice(non_successors)
    else:
        candidate = rng.choice(reachable)
    answer = "yes" if candidate in reachable else "no"

    alias_facts = render_alias_facts(graph)
    edge_facts = render_edge_facts(graph)
    rng.shuffle(alias_facts)
    rng.shuffle(edge_facts)
    walk_words = [_alias(graph, node, 1 + idx) for idx, node in enumerate(walk)]
    candidate_word = _alias(graph, candidate, 2)
    walk_text = " then ".join(walk_words)
    candidate_label = "Candidate next" if query_steps == 1 else f"Candidate after {query_steps} steps"
    prefix = (
        " ".join(alias_facts + edge_facts)
        + f" Walk: {walk_text}. {candidate_label}: {candidate_word}. Answer"
    )
    full_text = f"{prefix} {answer}."
    return GraphAliasExample(
        prefix=prefix,
        answer=answer,
        full_text=full_text,
        graph_id=graph.graph_id,
        current_node=current,
        candidate_node=candidate,
    )


class GraphAliasDataset(Dataset):
    """Balanced yes/no aliased graph-walk questions."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        graph_seed: int | None = None,
        edge_seed: int | None = None,
        example_seed: int | None = None,
        alias_pool_size: int | None = None,
        alias_pool_seed: int | None = None,
        n_graphs: int = 16,
        n_nodes: int = 16,
        aliases_per_node: int = 3,
        out_degree: int = 2,
        walk_length: int = 4,
        query_steps: int = 1,
    ) -> None:
        graph_seed = seed if graph_seed is None else graph_seed
        example_seed = seed if example_seed is None else example_seed
        alias_pool = None
        if alias_pool_size is not None:
            alias_pool = make_alias_pool(
                seed=graph_seed if alias_pool_seed is None else alias_pool_seed,
                size=alias_pool_size,
            )
        self.graphs = [
            generate_alias_graph(
                seed=graph_seed + 10_000 + graph_id,
                graph_id=f"ag{graph_id}",
                edge_seed=None if edge_seed is None else edge_seed + 20_000 + graph_id,
                alias_pool=alias_pool,
                n_nodes=n_nodes,
                aliases_per_node=aliases_per_node,
                out_degree=out_degree,
            )
            for graph_id in range(n_graphs)
        ]
        self.examples = []
        for idx in range(n_examples):
            graph = self.graphs[idx % n_graphs]
            self.examples.append(
                render_example(
                    graph,
                    seed=example_seed + idx,
                    walk_length=walk_length,
                    query_steps=query_steps,
                    positive=idx % 2 == 0,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> GraphAliasExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[GraphAliasExample]) -> list[GraphAliasExample]:
    return batch


if __name__ == "__main__":
    ds = GraphAliasDataset(n_examples=4, seed=1, n_graphs=1, n_nodes=6)
    for ex in ds:
        print(ex.full_text)
