"""Exact finite-automata equivalence tasks.

DFA language equivalence is a compact behavioral-equivalence benchmark with a
standard product-automaton verifier.  It is useful here because surface form can
be scrambled by state renaming while the accepted language remains unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset


ALPHABET = ("a", "b")


@dataclass(frozen=True)
class DFA:
    n_states: int
    start: int
    accept: frozenset[int]
    transitions: tuple[tuple[int, int], ...]  # state -> (on a, on b)


@dataclass(frozen=True)
class AutomataExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    positive: bool
    left: str
    right: str
    task_kind: str
    metadata: dict[str, str | int | bool]


def random_dfa(rng: random.Random, *, n_states: int) -> DFA:
    transitions = tuple(
        (rng.randrange(n_states), rng.randrange(n_states))
        for _ in range(n_states)
    )
    accept = frozenset(state for state in range(n_states) if rng.random() < 0.45)
    if not accept:
        accept = frozenset({rng.randrange(n_states)})
    return DFA(n_states=n_states, start=0, accept=accept, transitions=transitions)


def permute_dfa(dfa: DFA, rng: random.Random) -> DFA:
    perm = list(range(dfa.n_states))
    rng.shuffle(perm)
    # Keep start readable but still rename everything else.
    if perm[dfa.start] != 0:
        zero_at = perm.index(0)
        start_at = dfa.start
        perm[zero_at], perm[start_at] = perm[start_at], perm[zero_at]
    inv = {old: new for new, old in enumerate(perm)}
    transitions = []
    for old in perm:
        a, b = dfa.transitions[old]
        transitions.append((inv[a], inv[b]))
    accept = frozenset(inv[state] for state in dfa.accept)
    return DFA(n_states=dfa.n_states, start=inv[dfa.start], accept=accept, transitions=tuple(transitions))


def perturb_dfa(dfa: DFA, rng: random.Random) -> DFA:
    transitions = [list(row) for row in dfa.transitions]
    accept = set(dfa.accept)
    if rng.random() < 0.5:
        state = rng.randrange(dfa.n_states)
        if state in accept:
            accept.remove(state)
        else:
            accept.add(state)
        if not accept:
            accept.add((state + 1) % dfa.n_states)
    else:
        state = rng.randrange(dfa.n_states)
        symbol = rng.randrange(len(ALPHABET))
        old = transitions[state][symbol]
        choices = [s for s in range(dfa.n_states) if s != old]
        transitions[state][symbol] = rng.choice(choices)
    return DFA(
        n_states=dfa.n_states,
        start=dfa.start,
        accept=frozenset(accept),
        transitions=tuple((row[0], row[1]) for row in transitions),
    )


def equivalent(left: DFA, right: DFA) -> bool:
    agenda = [(left.start, right.start)]
    seen = set(agenda)
    while agenda:
        l_state, r_state = agenda.pop()
        if (l_state in left.accept) != (r_state in right.accept):
            return False
        for symbol_idx in range(len(ALPHABET)):
            pair = (
                left.transitions[l_state][symbol_idx],
                right.transitions[r_state][symbol_idx],
            )
            if pair not in seen:
                seen.add(pair)
                agenda.append(pair)
    return True


def render_dfa(dfa: DFA) -> str:
    accepts = " ".join(f"q{state}" for state in sorted(dfa.accept))
    parts = [f"start q{dfa.start}", f"accept {accepts}"]
    for state, (on_a, on_b) in enumerate(dfa.transitions):
        parts.append(f"q{state} a->q{on_a} b->q{on_b}")
    return " ; ".join(parts)


def render_dfa_equivalence_example(
    *,
    seed: int,
    graph_id: str,
    positive: bool,
    n_states: int = 6,
) -> AutomataExample:
    rng = random.Random(seed)
    left = random_dfa(rng, n_states=n_states)
    if positive:
        right = permute_dfa(left, rng)
        same = equivalent(left, right)
    else:
        right = left
        same = True
        for _ in range(100):
            candidate = permute_dfa(perturb_dfa(left, rng), rng)
            if not equivalent(left, candidate):
                right = candidate
                same = False
                break
        if same:
            right = DFA(
                n_states=left.n_states,
                start=left.start,
                accept=frozenset(set(range(left.n_states)) - set(left.accept)) or frozenset({0}),
                transitions=left.transitions,
            )
            same = equivalent(left, right)
    left_text = render_dfa(left)
    right_text = render_dfa(right)
    prefix = (
        f"Automaton A: {left_text}. Automaton B: {right_text}. "
        "Do these automata accept exactly the same strings? Answer"
    )
    answer = "yes" if same else "no"
    return AutomataExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        positive=same,
        left=left_text,
        right=right_text,
        task_kind="dfa_equivalence",
        metadata={"n_states": n_states},
    )


class DFAEquivalenceDataset(Dataset):
    """Balanced DFA language-equivalence examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        depth: int = 6,
    ) -> None:
        self.examples = [
            render_dfa_equivalence_example(
                seed=seed + idx,
                graph_id=f"dfaeq{idx}",
                positive=idx % 2 == 0,
                n_states=max(2, depth),
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> AutomataExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[AutomataExample]) -> list[AutomataExample]:
    return batch
