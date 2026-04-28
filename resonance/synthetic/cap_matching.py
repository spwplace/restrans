"""Cap-matching structural probe.

This is a self-contained experimental fragment inspired by the cap-matching
paper and the `reu_unif` reference implementation.  It tests whether a model
can decide if a first-order pattern can match at least one ground term denoted
by a Cap-term.

The full paper asks for a complete finite representation of all substitutions.
For the neural probe we deliberately use the simpler yes/no decision problem,
because it gives labels we can verify exactly while still requiring the model to
reason about variables, unions, constructors, and unbounded Cap closure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset


FUNCTIONS: tuple[tuple[str, int], ...] = (("p", 2), ("e", 2), ("f", 2), ("g", 1), ("h", 1))
CONSTANTS = ("a", "b", "c", "k")
VARIABLES = ("x", "y", "z", "m")


@dataclass(frozen=True)
class Term:
    name: str
    args: tuple["CapTerm", ...] = ()
    is_var: bool = False


@dataclass(frozen=True)
class Cap:
    seed: "CapTerm"
    constructors: tuple[str, ...]


@dataclass(frozen=True)
class UnionTerm:
    options: tuple["CapTerm", ...]


CapTerm = Term | Cap | UnionTerm
Subst = dict[str, CapTerm]


@dataclass(frozen=True)
class CapMatchingExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    matchable: bool
    pattern: str
    cap_term: str


def const(name: str) -> Term:
    return Term(name)


def var(name: str) -> Term:
    return Term(name, is_var=True)


def fun(name: str, *args: CapTerm) -> Term:
    return Term(name, tuple(args))


ARITY = {name: arity for name, arity in FUNCTIONS}


def render(term: CapTerm) -> str:
    if isinstance(term, Cap):
        constructors = " ".join(term.constructors)
        return f"Cap{{ {constructors} }} ( {render(term.seed)} )"
    if isinstance(term, UnionTerm):
        return " | ".join(render(option) for option in term.options)
    if not term.args:
        return term.name
    return f"{term.name} ( {' , '.join(render(arg) for arg in term.args)} )"


def term_depth(term: Term) -> int:
    if not term.args:
        return 0
    return 1 + max(term_depth(arg) for arg in term.args if isinstance(arg, Term))


def term_is_ground(term: CapTerm) -> bool:
    if isinstance(term, Term):
        return not term.is_var and all(term_is_ground(arg) for arg in term.args)
    return False


def cap_nonempty(term: CapTerm) -> bool:
    if isinstance(term, UnionTerm):
        return any(cap_nonempty(option) for option in term.options)
    if isinstance(term, Cap):
        return cap_nonempty(term.seed)
    return not term.is_var


def cap_intersects(left: CapTerm, right: CapTerm, fuel: int = 64) -> bool:
    """Conservative exact-enough intersection check for generated Cap fragments."""

    if fuel <= 0:
        return True
    if isinstance(left, UnionTerm):
        return any(cap_intersects(option, right, fuel - 1) for option in left.options)
    if isinstance(right, UnionTerm):
        return any(cap_intersects(left, option, fuel - 1) for option in right.options)
    if isinstance(left, Cap):
        if cap_intersects(left.seed, right, fuel - 1):
            return True
        if isinstance(right, Term) and not right.is_var and right.name in left.constructors:
            return all(cap_intersects(left, arg, fuel - 1) for arg in right.args)
        if isinstance(right, Cap):
            if cap_intersects(left.seed, right.seed, fuel - 1):
                return True
            return bool(set(left.constructors) & set(right.constructors)) and cap_nonempty(left) and cap_nonempty(right)
        return False
    if isinstance(right, Cap):
        return cap_intersects(right, left, fuel - 1)
    if left.is_var or right.is_var:
        return True
    if left.name != right.name or len(left.args) != len(right.args):
        return False
    return all(cap_intersects(l_arg, r_arg, fuel - 1) for l_arg, r_arg in zip(left.args, right.args))


def bind_var(name: str, region: CapTerm, subst: Subst) -> Subst | None:
    old = subst.get(name)
    if old is None:
        out = dict(subst)
        out[name] = region
        return out
    return subst if cap_intersects(old, region) else None


def match_cap(pattern: Term, cap_term: CapTerm, subst: Subst | None = None, fuel: int = 64) -> Subst | None:
    """Decide pattern-to-Cap matching without enumerating the Cap closure."""

    if fuel <= 0:
        return None
    subst = {} if subst is None else dict(subst)
    if pattern.is_var:
        return bind_var(pattern.name, cap_term, subst)
    if isinstance(cap_term, UnionTerm):
        for option in cap_term.options:
            out = match_cap(pattern, option, subst, fuel - 1)
            if out is not None:
                return out
        return None
    if isinstance(cap_term, Cap):
        from_seed = match_cap(pattern, cap_term.seed, subst, fuel - 1)
        if from_seed is not None:
            return from_seed
        if pattern.name not in cap_term.constructors:
            return None
        out = dict(subst)
        for arg in pattern.args:
            if not isinstance(arg, Term):
                return None
            out = match_cap(arg, cap_term, out, fuel - 1)
            if out is None:
                return None
        return out
    if cap_term.is_var or pattern.name != cap_term.name or len(pattern.args) != len(cap_term.args):
        return None
    out = dict(subst)
    for left, right in zip(pattern.args, cap_term.args):
        if not isinstance(left, Term):
            return None
        out = match_cap(left, right, out, fuel - 1)
        if out is None:
            return None
    return out


def cap_match_exists(pattern: Term, cap_term: CapTerm, max_depth: int = 5) -> bool:
    del max_depth
    return match_cap(pattern, cap_term) is not None


def random_ground(rng: random.Random, depth: int) -> Term:
    if depth <= 0 or rng.random() < 0.35:
        return const(rng.choice(CONSTANTS))
    name, arity = rng.choice(FUNCTIONS)
    return fun(name, *(random_ground(rng, depth - 1) for _ in range(arity)))


def random_pattern(rng: random.Random, depth: int, var_prob: float = 0.35) -> Term:
    if rng.random() < var_prob:
        return var(rng.choice(VARIABLES))
    if depth <= 0 or rng.random() < 0.25:
        return const(rng.choice(CONSTANTS))
    name, arity = rng.choice(FUNCTIONS)
    return fun(name, *(random_pattern(rng, depth - 1, var_prob) for _ in range(arity)))


def random_cap(rng: random.Random, depth: int) -> CapTerm:
    seed_options = tuple(random_ground(rng, max(0, depth - 2)) for _ in range(rng.randint(1, 3)))
    seed: CapTerm = seed_options[0] if len(seed_options) == 1 else UnionTerm(seed_options)
    constructors = tuple(sorted({rng.choice(FUNCTIONS)[0] for _ in range(rng.randint(1, 3))}))
    cap: CapTerm = Cap(seed, constructors)
    if rng.random() < 0.4:
        cap = UnionTerm((cap, random_ground(rng, depth=1)))
    if rng.random() < 0.35:
        wrapper, arity = rng.choice(FUNCTIONS)
        if arity == 1:
            cap = Term(wrapper, (cap,))  # type: ignore[arg-type]
        else:
            cap = Term(wrapper, (cap, random_ground(rng, depth=1)))  # type: ignore[arg-type]
    return cap


def sample_from_cap(rng: random.Random, cap_term: CapTerm, depth: int) -> Term:
    if isinstance(cap_term, UnionTerm):
        return sample_from_cap(rng, rng.choice(cap_term.options), depth)
    if isinstance(cap_term, Cap):
        if depth <= 0 or rng.random() < 0.45:
            return sample_from_cap(rng, cap_term.seed, depth)
        name = rng.choice(cap_term.constructors)
        arity = ARITY[name]
        return fun(name, *(sample_from_cap(rng, cap_term, depth - 1) for _ in range(arity)))
    if cap_term.is_var:
        raise ValueError("cannot sample a ground term from a variable")
    return fun(cap_term.name, *(sample_from_cap(rng, arg, depth - 1) for arg in cap_term.args))


def positive_pair(rng: random.Random, depth: int) -> tuple[Term, CapTerm]:
    cap = random_cap(rng, depth)
    target = sample_from_cap(rng, cap, depth=max(1, depth))
    return mask_target(rng, target), cap


def mask_target(rng: random.Random, target: Term, var_prob: float = 0.35) -> Term:
    if rng.random() < var_prob:
        return var(rng.choice(VARIABLES))
    if not target.args:
        return target
    return fun(target.name, *(mask_target(rng, arg, var_prob) for arg in target.args))


def negative_pair(rng: random.Random, depth: int) -> tuple[Term, CapTerm]:
    for _ in range(1_000):
        pattern = random_pattern(rng, depth)
        cap = random_cap(rng, depth)
        if not cap_match_exists(pattern, cap, max_depth=max(4, depth + 1)):
            return pattern, cap
    return fun("p", const("a"), const("b")), Cap(const("k"), ("g", "h"))


def render_cap_matching_example(seed: int, graph_id: str, depth: int, positive: bool) -> CapMatchingExample:
    rng = random.Random(seed)
    pattern, cap_term = positive_pair(rng, depth) if positive else negative_pair(rng, depth)
    matchable = cap_match_exists(pattern, cap_term, max_depth=max(4, depth + 1))
    pattern_text = render(pattern)
    cap_text = render(cap_term)
    prefix = f"Pattern term: {pattern_text}. Cap object: {cap_text}. Does the pattern match some represented term? Answer"
    answer = "yes" if matchable else "no"
    return CapMatchingExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        matchable=matchable,
        pattern=pattern_text,
        cap_term=cap_text,
    )


class CapMatchingDataset(Dataset):
    """Balanced yes/no cap-matching examples."""

    def __init__(self, *, n_examples: int = 512, seed: int = 0, depth: int = 4) -> None:
        self.examples = [
            render_cap_matching_example(
                seed=seed + idx,
                graph_id=f"cap{idx}",
                depth=depth,
                positive=idx % 2 == 0,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> CapMatchingExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[CapMatchingExample]) -> list[CapMatchingExample]:
    return batch
