"""First-order term unification task.

Examples ask whether two first-order terms have a unifier.  Positives are made
by applying a random substitution to one side.  Negatives are generated and
verified with an occurs-check unifier, so labels are not heuristic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset


FUNCTIONS: tuple[tuple[str, int], ...] = (
    ("f", 2),
    ("g", 2),
    ("h", 1),
    ("p", 1),
)
CONSTANTS = ("a", "b", "c", "d")
VARIABLES = ("x", "y", "z", "u", "v", "w")


@dataclass(frozen=True)
class Term:
    pass


@dataclass(frozen=True)
class Var(Term):
    name: str


@dataclass(frozen=True)
class Fun(Term):
    name: str
    args: tuple[Term, ...] = ()


Subst = dict[str, Term]


@dataclass(frozen=True)
class UnificationExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    unifiable: bool
    left: str
    right: str


def render(term: Term) -> str:
    if isinstance(term, Var):
        return term.name
    if not term.args:
        return term.name
    return f"{term.name} ( {' , '.join(render(arg) for arg in term.args)} )"


def free_vars(term: Term) -> set[str]:
    if isinstance(term, Var):
        return {term.name}
    return set().union(*(free_vars(arg) for arg in term.args)) if term.args else set()


def apply_subst(term: Term, subst: Subst, seen: frozenset[str] = frozenset()) -> Term:
    if isinstance(term, Var):
        if term.name in subst:
            if term.name in seen:
                return term
            return apply_subst(subst[term.name], subst, seen | {term.name})
        return term
    return Fun(term.name, tuple(apply_subst(arg, subst, seen) for arg in term.args))


def occurs(name: str, term: Term, subst: Subst) -> bool:
    term = apply_subst(term, subst)
    if isinstance(term, Var):
        return term.name == name
    return any(occurs(name, arg, subst) for arg in term.args)


def unify(left: Term, right: Term, subst: Subst | None = None) -> Subst | None:
    subst = {} if subst is None else dict(subst)
    left = apply_subst(left, subst)
    right = apply_subst(right, subst)
    if left == right:
        return subst
    if isinstance(left, Var):
        if occurs(left.name, right, subst):
            return None
        subst[left.name] = right
        return subst
    if isinstance(right, Var):
        if occurs(right.name, left, subst):
            return None
        subst[right.name] = left
        return subst
    if left.name != right.name or len(left.args) != len(right.args):
        return None
    for left_arg, right_arg in zip(left.args, right.args):
        subst = unify(left_arg, right_arg, subst)
        if subst is None:
            return None
    return subst


def is_unifiable(left: Term, right: Term) -> bool:
    return unify(left, right) is not None


def random_term(
    rng: random.Random,
    *,
    depth: int,
    var_prob: float = 0.35,
) -> Term:
    if depth <= 0:
        if rng.random() < var_prob:
            return Var(rng.choice(VARIABLES))
        return Fun(rng.choice(CONSTANTS))
    if rng.random() < var_prob:
        return Var(rng.choice(VARIABLES))
    if rng.random() < 0.25:
        return Fun(rng.choice(CONSTANTS))
    name, arity = rng.choice(FUNCTIONS)
    return Fun(name, tuple(random_term(rng, depth=depth - 1, var_prob=var_prob) for _ in range(arity)))


def random_substitution(rng: random.Random, variables: set[str], depth: int) -> Subst:
    subst: Subst = {}
    for name in sorted(variables):
        if rng.random() < 0.65:
            replacement = random_term(rng, depth=max(0, depth - 1), var_prob=0.15)
            if occurs(name, replacement, subst):
                replacement = Fun(rng.choice(CONSTANTS))
            subst[name] = replacement
    return subst


def render_unification_example(
    *,
    seed: int,
    graph_id: str,
    depth: int = 4,
    positive: bool = True,
) -> UnificationExample:
    rng = random.Random(seed)
    if positive:
        left = random_term(rng, depth=depth)
        subst = random_substitution(rng, free_vars(left), depth)
        right = apply_subst(left, subst)
        unifiable = True
    else:
        left = right = random_term(rng, depth=depth)
        for _ in range(1_000):
            left = random_term(rng, depth=depth)
            right = random_term(rng, depth=depth)
            if not is_unifiable(left, right):
                break
        unifiable = is_unifiable(left, right)
        if unifiable:
            left = Fun("f", (Fun("a"), Fun("b")))
            right = Fun("g", (Fun("a"), Fun("b")))
            unifiable = False

    left_text = render(left)
    right_text = render(right)
    prefix = f"Term one: {left_text}. Term two: {right_text}. Do the terms unify? Answer"
    answer = "yes" if unifiable else "no"
    return UnificationExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        unifiable=unifiable,
        left=left_text,
        right=right_text,
    )


class UnificationDataset(Dataset):
    """Balanced yes/no first-order unification examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        depth: int = 4,
    ) -> None:
        self.examples = [
            render_unification_example(
                seed=seed + idx,
                graph_id=f"uni{idx}",
                depth=depth,
                positive=idx % 2 == 0,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> UnificationExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[UnificationExample]) -> list[UnificationExample]:
    return batch
