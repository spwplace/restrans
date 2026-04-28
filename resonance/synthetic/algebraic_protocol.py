"""Algebraic protocol reachability probe.

This synthetic task models a small Dolev-Yao-style intruder.  Given initial
knowledge and constructor/destructor abilities, decide whether a query message
is derivable.  It is deliberately bounded for dataset generation, but the
examples expose the same algebraic structure that motivates cap matching:
pairing, encryption, projection, and decryption with a known key.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset

from .cap_matching import Term, const, fun, render


ATOMS = ("alice", "bob", "eve", "n1", "n2", "secret", "k_ab", "k_e")


@dataclass(frozen=True)
class AlgebraicProtocolExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    reachable: bool
    query: str


def depth(term: Term) -> int:
    if not term.args:
        return 0
    return 1 + max(depth(arg) for arg in term.args if isinstance(arg, Term))


def closure(initial: set[Term], *, max_depth: int = 3, max_terms: int = 220) -> set[Term]:
    known = set(initial)
    changed = True
    while changed and len(known) < max_terms:
        changed = False
        current = sorted(known, key=render)

        # Destructors: p(x,y) -> x,y and e(m,k),k -> m.
        for term in current:
            if term.name == "p" and len(term.args) == 2:
                for arg in term.args:
                    if isinstance(arg, Term) and arg not in known:
                        known.add(arg)
                        changed = True
            if term.name == "e" and len(term.args) == 2 and isinstance(term.args[0], Term) and isinstance(term.args[1], Term):
                if term.args[1] in known and term.args[0] not in known:
                    known.add(term.args[0])
                    changed = True

        current = sorted(known, key=render)
        small = [term for term in current if depth(term) < max_depth]
        for left in small[:64]:
            for right in small[:64]:
                for candidate in (fun("p", left, right), fun("e", left, right)):
                    if depth(candidate) <= max_depth and candidate not in known:
                        known.add(candidate)
                        changed = True
                        if len(known) >= max_terms:
                            break
                if len(known) >= max_terms:
                    break
            if len(known) >= max_terms:
                break
    return known


def random_initial(rng: random.Random) -> set[Term]:
    atoms = [const(atom) for atom in rng.sample(ATOMS, rng.randint(3, 5))]
    known = set(atoms)
    if rng.random() < 0.7:
        known.add(fun("p", rng.choice(atoms), rng.choice(atoms)))
    if rng.random() < 0.7:
        message = rng.choice(atoms)
        key = rng.choice([const("k_ab"), const("k_e")])
        known.add(fun("e", message, key))
    return known


def random_term(rng: random.Random, atom_pool: list[Term], max_depth: int) -> Term:
    if max_depth <= 0 or rng.random() < 0.45:
        return rng.choice(atom_pool)
    name = rng.choice(("p", "e"))
    return fun(name, random_term(rng, atom_pool, max_depth - 1), random_term(rng, atom_pool, max_depth - 1))


def render_example(seed: int, graph_id: str, positive: bool, max_depth: int) -> AlgebraicProtocolExample:
    rng = random.Random(seed)
    initial = random_initial(rng)
    known = closure(initial, max_depth=max_depth)
    if positive:
        query = rng.choice(sorted(known, key=render))
        reachable = True
    else:
        atoms = [const(atom) for atom in ATOMS]
        query = random_term(rng, atoms, max_depth)
        for _ in range(500):
            if query not in known:
                break
            query = random_term(rng, atoms, max_depth)
        reachable = query in known
        if reachable:
            query = fun("e", const("secret"), const("k_unknown"))
            reachable = False

    facts = " ; ".join(render(term) for term in sorted(initial, key=render))
    abilities = "construct pair p(x,y); construct encryption e(m,k); split p(x,y); decrypt e(m,k) when k is known"
    prefix = f"Known messages: {facts}. Intruder abilities: {abilities}. Query message: {render(query)}. Can the intruder derive the query? Answer"
    answer = "yes" if reachable else "no"
    return AlgebraicProtocolExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        reachable=reachable,
        query=render(query),
    )


class AlgebraicProtocolDataset(Dataset):
    """Balanced yes/no Dolev-Yao reachability examples."""

    def __init__(self, *, n_examples: int = 512, seed: int = 0, depth: int = 3) -> None:
        self.examples = [
            render_example(seed + idx, f"algproto{idx}", positive=idx % 2 == 0, max_depth=depth)
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> AlgebraicProtocolExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[AlgebraicProtocolExample]) -> list[AlgebraicProtocolExample]:
    return batch
