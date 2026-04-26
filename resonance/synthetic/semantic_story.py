"""Verified semantic-graph story generator.

This module bridges formal equivalence and natural language with a small,
typed event-graph language.  Each graph can be rendered as several surface
stories, while hard negatives are produced by mutating one structural edge or
predicate argument.

The generator is intentionally simple.  Its job is not literary quality; it is
to provide topology-bearing natural language where the ground-truth structure is
known and falsifiable.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

import torch
from torch.utils.data import Dataset


class Relation(str, Enum):
    BEFORE = "before"
    CAUSES = "causes"
    ENABLES = "enables"
    PREVENTS = "prevents"
    BELIEVES = "believes"
    WANTS = "wants"


@dataclass(frozen=True)
class Entity:
    name: str
    kind: str


@dataclass(frozen=True)
class Event:
    event_id: str
    predicate: str
    subject: str
    obj: str | None = None
    target: str | None = None
    location: str | None = None
    negated: bool = False
    quantifier: str | None = None

    def signature(self) -> str:
        parts = [
            self.event_id,
            self.predicate,
            self.subject,
            self.obj or "_",
            self.target or "_",
            self.location or "_",
            "not" if self.negated else "yes",
            self.quantifier or "_",
        ]
        return ":".join(parts)


@dataclass(frozen=True)
class Edge:
    relation: Relation
    source: str
    target: str

    def signature(self) -> str:
        return f"{self.relation.value}:{self.source}->{self.target}"


@dataclass(frozen=True)
class StoryGraph:
    graph_id: str
    entities: tuple[Entity, ...]
    events: tuple[Event, ...]
    edges: tuple[Edge, ...]

    def signature(self) -> str:
        entities = ";".join(sorted(f"{e.name}:{e.kind}" for e in self.entities))
        events = ";".join(sorted(e.signature() for e in self.events))
        edges = ";".join(sorted(e.signature() for e in self.edges))
        return f"E[{entities}]|V[{events}]|R[{edges}]"


NAMES = [
    "Mira",
    "Jon",
    "Lea",
    "Sam",
    "Nia",
    "Omar",
    "Tess",
    "Rin",
]
OBJECTS = ["map", "key", "lamp", "book", "coin", "box", "rope", "cup"]
PLACES = ["garden", "library", "market", "tower", "kitchen", "workshop"]
PREDICATES = ["finds", "gives", "hides", "moves", "repairs", "carries"]

VERB_FORMS = {
    "finds": ("found", "find", "found"),
    "gives": ("gave", "give", "given"),
    "hides": ("hid", "hide", "hidden"),
    "moves": ("moved", "move", "moved"),
    "repairs": ("repaired", "repair", "repaired"),
    "carries": ("carried", "carry", "carried"),
}


def _choice(rng: random.Random, values: list[str]) -> str:
    return values[rng.randrange(len(values))]


def generate_story_graph(seed: int, graph_id: str | None = None) -> StoryGraph:
    """Generate a small event graph with temporal and causal structure."""
    rng = random.Random(seed)
    people = rng.sample(NAMES, 3)
    obj_a, obj_b = rng.sample(OBJECTS, 2)
    place = _choice(rng, PLACES)
    pred_a, pred_b = rng.sample(PREDICATES, 2)

    entities = (
        Entity(people[0], "person"),
        Entity(people[1], "person"),
        Entity(people[2], "person"),
        Entity(obj_a, "object"),
        Entity(obj_b, "object"),
        Entity(place, "place"),
    )

    quant_a = rng.choice([None, "all", "some", "none"]) if rng.random() < 0.5 else None
    quant_b = rng.choice([None, "all", "some", "none"]) if rng.random() < 0.5 else None

    e1 = Event("e1", pred_a, people[0], obj=obj_a, location=place, quantifier=quant_a)
    e2 = Event("e2", "gives", people[0], obj=obj_a, target=people[1], location=place)
    e3 = Event("e3", pred_b, people[1], obj=obj_b, location=place, quantifier=quant_b)
    e4 = Event("e4", "finds", people[2], obj=obj_b, location=place)

    # Keep one stable temporal backbone and vary one semantic relation.
    relation = rng.choice([Relation.CAUSES, Relation.ENABLES, Relation.PREVENTS])
    edges = (
        Edge(Relation.BEFORE, "e1", "e2"),
        Edge(relation, "e2", "e3"),
        Edge(Relation.BEFORE, "e3", "e4"),
        Edge(Relation.BELIEVES, people[2], "e3"),
    )
    return StoryGraph(graph_id or f"g{seed}", entities, (e1, e2, e3, e4), edges)


def event_sentence(event: Event, style: str = "plain") -> str:
    """Render a single event as English."""
    verb_forms = VERB_FORMS[event.predicate]
    past = verb_forms[0]
    infinitive = verb_forms[1]
    participle = verb_forms[2] if len(verb_forms) > 2 else past
    neg = "did not " if event.negated else ""
    verb = infinitive if event.negated else past

    obj_phrase = ""
    if event.obj:
        if event.quantifier == "all":
            obj_phrase = f" every {event.obj}"
        elif event.quantifier == "some":
            obj_phrase = f" a {event.obj}"
        elif event.quantifier == "none":
            obj_phrase = f" no {event.obj}"
        else:
            obj_phrase = f" the {event.obj}"

    target = f" to {event.target}" if event.target else ""
    location = f" in the {event.location}" if event.location else ""

    if style == "passive" and event.obj:
        obj_text = obj_phrase.strip()
        if event.negated:
            return f"{obj_text} was not {participle} by {event.subject}{target}{location}"
        return f"{obj_text} was {participle} by {event.subject}{target}{location}"

    if style == "compact":
        location = ""

    return f"{event.subject} {neg}{verb}{obj_phrase}{target}{location}"


def edge_clause(edge: Edge, event_map: dict[str, Event], variant: int = 0) -> str:
    """Render an edge as a discourse connective clause."""
    if edge.relation == Relation.BEFORE:
        choices = ["After that", "Before that", "Later", "Soon", "Then"]
        return choices[variant % len(choices)]
    if edge.relation == Relation.CAUSES:
        choices = ["Because of that", "Therefore", "Consequently", "As a result"]
        return choices[variant % len(choices)]
    if edge.relation == Relation.ENABLES:
        choices = ["That made it possible that", "This allowed", "Thanks to that"]
        return choices[variant % len(choices)]
    if edge.relation == Relation.PREVENTS:
        choices = ["Even so", "Nevertheless", "Despite that", "Still"]
        return choices[variant % len(choices)]
    if edge.relation == Relation.BELIEVES:
        event = event_map.get(edge.target)
        if event is not None:
            choices = [
                f"{edge.source} thought that {event_sentence(event, 'compact')}",
                f"{edge.source} believed that {event_sentence(event, 'compact')}",
                f"{edge.source} was sure that {event_sentence(event, 'compact')}",
            ]
            return choices[variant % len(choices)]
    if edge.relation == Relation.WANTS:
        event = event_map.get(edge.target)
        if event is not None:
            choices = [
                f"{edge.source} wanted {event_sentence(event, 'compact')}",
                f"{edge.source} wished that {event_sentence(event, 'compact')}",
                f"{edge.source} desired {event_sentence(event, 'compact')}",
            ]
            return choices[variant % len(choices)]
    return "Then"


def render_story(graph: StoryGraph, variant: int = 0) -> str:
    """Render the same graph with controlled surface variation."""
    event_map = {event.event_id: event for event in graph.events}
    events = list(graph.events)
    if variant % 5 == 1:
        place = next((e.name for e in graph.entities if e.kind == "place"), "place")
        prefix = f"In the {place}, something happened."
    elif variant % 5 == 2:
        people = [e.name for e in graph.entities if e.kind == "person"]
        prefix = f"{people[0]}, {people[1]}, and {people[2]} were together."
    elif variant % 5 == 3:
        prefix = "A series of events unfolded."
    elif variant % 5 == 4:
        people = [e.name for e in graph.entities if e.kind == "person"]
        prefix = f"It began with {people[0]}."
    else:
        prefix = ""

    sentences: list[str] = []
    if prefix:
        sentences.append(prefix)

    edge_by_target = {
        edge.target: edge
        for edge in graph.edges
        if edge.source.startswith("e") and edge.target.startswith("e")
    }
    for idx, event in enumerate(events):
        # Mix surface styles per event to increase lexical overlap with
        # topology-breaking negatives.
        event_variant = (variant + idx) % 4
        if event_variant == 0:
            style = "plain"
        elif event_variant == 1:
            style = "compact"
        elif event_variant == 2 and event.obj:
            style = "passive"
        else:
            style = "plain"

        sentence = event_sentence(event, style=style)
        edge = edge_by_target.get(event.event_id)
        if edge and idx > 0:
            connective = edge_clause(edge, event_map, variant=variant)
            sentence = f"{connective}, {sentence[0].lower()}{sentence[1:]}"
        sentences.append(sentence + ".")

    belief_edges = [edge for edge in graph.edges if edge.relation in {Relation.BELIEVES, Relation.WANTS}]
    for edge in belief_edges:
        clause = edge_clause(edge, event_map, variant=variant)
        if clause:
            sentences.append(clause + ".")

    if variant % 7 == 6:
        sentences.append("The order of these events mattered.")
    return " ".join(sentences)


# Alias for callers that expect the longer name
render_story_text = render_story


def mutate_graph(graph: StoryGraph, seed: int, mutation: str | None = None) -> StoryGraph:
    """Create a hard negative by changing one topological fact."""
    rng = random.Random(seed)
    mutation = mutation or rng.choice([
        "reverse_edge",
        "swap_target",
        "flip_relation",
        "negate_event",
        "swap_subject_object",
        "change_quantifier",
        "remove_negation",
        "change_location",
        "weaken_cause",
        "swap_belief_holder",
        "negate_belief_content",
    ])

    if mutation == "reverse_edge":
        edges = list(graph.edges)
        candidates = [i for i, edge in enumerate(edges)]
        if candidates:
            idx = rng.choice(candidates)
            edge = edges[idx]
            edges[idx] = Edge(edge.relation, edge.target, edge.source)
        return replace(graph, graph_id=f"{graph.graph_id}_rev", edges=tuple(edges))

    if mutation == "swap_target":
        events = list(graph.events)
        people = [entity.name for entity in graph.entities if entity.kind == "person"]
        candidates = [i for i, event in enumerate(events) if event.target is not None]
        if candidates:
            idx = rng.choice(candidates)
            event = events[idx]
            new_target = rng.choice([p for p in people if p != event.target])
            events[idx] = replace(event, target=new_target)
        return replace(graph, graph_id=f"{graph.graph_id}_target", events=tuple(events))

    if mutation == "flip_relation":
        edges = list(graph.edges)
        candidates = [i for i, edge in enumerate(edges) if edge.relation in {Relation.CAUSES, Relation.ENABLES, Relation.PREVENTS}]
        if candidates:
            idx = rng.choice(candidates)
            edge = edges[idx]
            choices = [r for r in (Relation.CAUSES, Relation.ENABLES, Relation.PREVENTS) if r != edge.relation]
            edges[idx] = Edge(rng.choice(choices), edge.source, edge.target)
        return replace(graph, graph_id=f"{graph.graph_id}_rel", edges=tuple(edges))

    if mutation == "negate_event":
        events = list(graph.events)
        idx = rng.randrange(len(events))
        events[idx] = replace(events[idx], negated=not events[idx].negated)
        return replace(graph, graph_id=f"{graph.graph_id}_neg", events=tuple(events))

    if mutation == "swap_subject_object":
        events = list(graph.events)
        candidates = [i for i, event in enumerate(events) if event.obj is not None]
        if candidates:
            idx = rng.choice(candidates)
            event = events[idx]
            events[idx] = replace(event, subject=event.obj, obj=event.subject)
        return replace(graph, graph_id=f"{graph.graph_id}_subjobj", events=tuple(events))

    if mutation == "change_quantifier":
        events = list(graph.events)
        candidates = [i for i, event in enumerate(events) if event.obj is not None]
        if candidates:
            idx = rng.choice(candidates)
            event = events[idx]
            cycle = {"all": "some", "some": "none", "none": "all", None: "some"}
            events[idx] = replace(event, quantifier=cycle.get(event.quantifier, "some"))
        return replace(graph, graph_id=f"{graph.graph_id}_quant", events=tuple(events))

    if mutation == "remove_negation":
        events = list(graph.events)
        candidates = [i for i, event in enumerate(events) if event.negated]
        if candidates:
            idx = rng.choice(candidates)
            events[idx] = replace(events[idx], negated=False)
        else:
            idx = rng.randrange(len(events))
            events[idx] = replace(events[idx], negated=not events[idx].negated)
        return replace(graph, graph_id=f"{graph.graph_id}_remneg", events=tuple(events))

    if mutation == "change_location":
        events = list(graph.events)
        candidates = [i for i, event in enumerate(events) if event.location is not None]
        if candidates:
            idx = rng.choice(candidates)
            event = events[idx]
            new_place = rng.choice([p for p in PLACES if p != event.location])
            events[idx] = replace(event, location=new_place)
        return replace(graph, graph_id=f"{graph.graph_id}_loc", events=tuple(events))

    if mutation == "weaken_cause":
        edges = list(graph.edges)
        candidates = [i for i, edge in enumerate(edges) if edge.relation in {Relation.CAUSES, Relation.PREVENTS}]
        if candidates:
            idx = rng.choice(candidates)
            edge = edges[idx]
            if edge.relation == Relation.CAUSES:
                edges[idx] = Edge(Relation.ENABLES, edge.source, edge.target)
            else:
                edges.pop(idx)
        return replace(graph, graph_id=f"{graph.graph_id}_weak", edges=tuple(edges))

    if mutation == "swap_belief_holder":
        edges = list(graph.edges)
        candidates = [i for i, edge in enumerate(edges) if edge.relation in {Relation.BELIEVES, Relation.WANTS}]
        if candidates:
            idx = rng.choice(candidates)
            edge = edges[idx]
            people = [entity.name for entity in graph.entities if entity.kind == "person"]
            new_source = rng.choice([p for p in people if p != edge.source])
            edges[idx] = Edge(edge.relation, new_source, edge.target)
        return replace(graph, graph_id=f"{graph.graph_id}_belief", edges=tuple(edges))

    if mutation == "negate_belief_content":
        edges = list(graph.edges)
        candidates = [i for i, edge in enumerate(edges) if edge.relation == Relation.BELIEVES]
        if candidates:
            idx = rng.choice(candidates)
            edge = edges[idx]
            events = list(graph.events)
            event_idx = next((i for i, e in enumerate(events) if e.event_id == edge.target), None)
            if event_idx is not None:
                events[event_idx] = replace(events[event_idx], negated=not events[event_idx].negated)
                return replace(graph, graph_id=f"{graph.graph_id}_negbel", events=tuple(events))
        return mutate_graph(graph, seed, mutation="negate_event")

    raise ValueError(f"unknown graph mutation: {mutation}")


def _graph_diff_stats(orig: StoryGraph, mutated: StoryGraph) -> dict[str, int]:
    """Count structural differences between two graphs."""
    stats = {"events": 0, "edges": 0, "entities": 0}

    e1_map = {e.event_id: e for e in orig.events}
    e2_map = {e.event_id: e for e in mutated.events}
    for eid in set(e1_map) | set(e2_map):
        s1 = e1_map.get(eid)
        s2 = e2_map.get(eid)
        if s1 is None or s2 is None or s1.signature() != s2.signature():
            stats["events"] += 1

    edges1 = {e.signature() for e in orig.edges}
    edges2 = {e.signature() for e in mutated.edges}
    stats["edges"] = len(edges1.symmetric_difference(edges2))

    ent1 = {f"{e.name}:{e.kind}" for e in orig.entities}
    ent2 = {f"{e.name}:{e.kind}" for e in mutated.entities}
    stats["entities"] = len(ent1.symmetric_difference(ent2))

    return stats


def verify_hard_negatives(n: int = 100, seed: int = 42) -> dict[str, object]:
    """Generate N graphs with positives and negatives and run sanity checks.

    Checks:
      - No negative text exactly equals any positive text.
      - Each negative graph differs from the original by a bounded structural
        change (at most 1 event or 2 edges).
      - Reports the empirical mutation type distribution.
    """
    dataset = StoryGraphDataset(
        n_graphs=n,
        variants_per_graph=4,
        hard_negatives_per_graph=2,
        seed=seed,
    )

    stats: dict[str, object] = {
        "total_graphs": n,
        "total_positives": 0,
        "total_negatives": 0,
        "positive_negative_collisions": 0,
        "valid_single_mutation": 0,
        "mutation_distribution": Counter(),
    }

    for row in dataset:
        graph = row["graph"]
        positives = row["positives"]
        negatives = row["hard_negatives"]
        mutations = row.get("mutations", [])
        neg_graphs = row.get("negative_graphs", [])

        stats["total_positives"] += len(positives)  # type: ignore[operator]
        stats["total_negatives"] += len(negatives)  # type: ignore[operator]

        for neg_text, neg_graph, mutation in zip(negatives, neg_graphs, mutations):
            if neg_text in positives:
                stats["positive_negative_collisions"] += 1  # type: ignore[operator]

            if neg_graph is not None:
                diff = _graph_diff_stats(graph, neg_graph)
                if diff["entities"] == 0 and (diff["events"] <= 1) and (diff["edges"] <= 2):
                    stats["valid_single_mutation"] += 1  # type: ignore[operator]

            if mutation:
                stats["mutation_distribution"][mutation] += 1  # type: ignore[index]

    return stats


class StoryTokenizer:
    """Small whitespace tokenizer for generated stories."""

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3
        self.word2id = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.id2word = {v: k for k, v in self.word2id.items()}

    @staticmethod
    def _words(text: str) -> list[str]:
        cleaned = text.lower()
        for ch in ".,;:!?":
            cleaned = cleaned.replace(ch, f" {ch} ")
        return cleaned.split()

    def train(self, texts: Iterable[str]) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for word in self._words(text):
                counts[word] = counts.get(word, 0) + 1
        for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if len(self.word2id) >= self.vocab_size:
                break
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.sos_id]
        ids.extend(self.word2id.get(word, self.unk_id) for word in self._words(text))
        ids.append(self.eos_id)
        ids = ids[:max_length]
        ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

    def batch_encode(self, texts: list[str], max_length: int, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.tensor(
            [self.encode(text, max_length) for text in texts],
            dtype=torch.long,
            device=device,
        )


class StoryGraphDataset(Dataset):
    """Dataset of same-graph story groups plus optional hard negatives."""

    ALL_MUTATIONS = [
        "reverse_edge",
        "swap_target",
        "flip_relation",
        "negate_event",
        "swap_subject_object",
        "change_quantifier",
        "remove_negation",
        "change_location",
        "weaken_cause",
        "swap_belief_holder",
        "negate_belief_content",
    ]

    def __init__(
        self,
        *,
        n_graphs: int = 128,
        variants_per_graph: int = 4,
        hard_negatives_per_graph: int = 2,
        seed: int = 0,
    ) -> None:
        self.rows: list[dict[str, object]] = []
        rng = random.Random(seed)
        for i in range(n_graphs):
            graph = generate_story_graph(seed + i, graph_id=f"g{i}")
            positives = [render_story(graph, variant=v) for v in range(variants_per_graph)]
            negative_texts: list[str] = []
            negative_graphs: list[StoryGraph] = []
            used_mutations: list[str] = []

            for j in range(hard_negatives_per_graph):
                variant = j % variants_per_graph
                positive_text = positives[variant]

                available = list(self.ALL_MUTATIONS)
                rng.shuffle(available)

                neg_text = positive_text
                neg_graph = graph
                mutation_used = "none"

                for mutation in available:
                    candidate_graph = mutate_graph(
                        graph,
                        seed + 10_000 + i * 17 + j + self.ALL_MUTATIONS.index(mutation),
                        mutation=mutation,
                    )
                    candidate_text = render_story(candidate_graph, variant=variant)
                    if candidate_text != positive_text:
                        neg_text = candidate_text
                        neg_graph = candidate_graph
                        mutation_used = mutation
                        break

                # Fallback: force a negation flip if nothing changed text.
                if neg_text == positive_text:
                    forced_graph = mutate_graph(
                        graph,
                        seed + 20_000 + i * 17 + j,
                        mutation="negate_event",
                    )
                    forced_text = render_story(forced_graph, variant=variant)
                    if forced_text != positive_text:
                        neg_text = forced_text
                        neg_graph = forced_graph
                        mutation_used = "negate_event"

                negative_texts.append(neg_text)
                negative_graphs.append(neg_graph)
                used_mutations.append(mutation_used)

            self.rows.append(
                {
                    "graph": graph,
                    "signature": graph.signature(),
                    "positives": positives,
                    "hard_negatives": negative_texts,
                    "negative_graphs": negative_graphs,
                    "mutations": used_mutations,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.rows[idx]

    def all_texts(self) -> list[str]:
        texts: list[str] = []
        for row in self.rows:
            texts.extend(row["positives"])  # type: ignore[arg-type]
            texts.extend(row["hard_negatives"])  # type: ignore[arg-type]
        return texts

    @staticmethod
    def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
        """Collate for contrastive training.

        Returns lists of positives, hard negatives, and structural metadata so
        that a contrastive loss can treat same-graph positives as anchors and
        hard negatives as distractors.
        """
        return {
            "positives": [row["positives"] for row in batch],
            "hard_negatives": [row["hard_negatives"] for row in batch],
            "signatures": [row["signature"] for row in batch],
            "graphs": [row["graph"] for row in batch],
            "negative_graphs": [row.get("negative_graphs", []) for row in batch],
            "mutations": [row.get("mutations", []) for row in batch],
        }


if __name__ == "__main__":
    dataset = StoryGraphDataset(n_graphs=2, variants_per_graph=3, hard_negatives_per_graph=2, seed=7)
    for row in dataset:
        print("SIGNATURE", row["signature"])
        print("POS")
        for text in row["positives"]:  # type: ignore[union-attr]
            print(" ", text)
        print("NEG")
        for text in row["hard_negatives"]:  # type: ignore[union-attr]
            print(" ", text)
        print("MUTATIONS", row.get("mutations"))

    print("\n--- Verification ---")
    result = verify_hard_negatives(n=50, seed=7)
    for key, value in result.items():
        print(f"{key}: {value}")
