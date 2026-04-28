"""ListOps-style parse/composition validation task.

The expression grammar is deliberately small but structurally nontrivial:

    [MAX 2 7 [MIN 4 1]]
    [SUM 1 [MAX 3 8] 2]

Examples ask whether the expression evaluates to a proposed digit.  The label
is exact and requires recovering the tree; local token counts and surface
lexemes are deliberately insufficient once depth increases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset


OPS = ("MAX", "MIN", "SUM", "PROD")


@dataclass(frozen=True)
class ListExpr:
    op: str | None
    value: int | None = None
    children: tuple["ListExpr", ...] = ()


@dataclass(frozen=True)
class ListOpsExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    correct: bool
    expression: str
    value: int
    candidate: int


def eval_expr(expr: ListExpr) -> int:
    if expr.op is None:
        assert expr.value is not None
        return expr.value
    values = [eval_expr(child) for child in expr.children]
    if expr.op == "MAX":
        return max(values)
    if expr.op == "MIN":
        return min(values)
    if expr.op == "SUM":
        return sum(values) % 10
    if expr.op == "PROD":
        out = 1
        for value in values:
            out = (out * value) % 10
        return out
    raise ValueError(f"unknown op: {expr.op}")


def render_expr(expr: ListExpr) -> str:
    if expr.op is None:
        assert expr.value is not None
        return str(expr.value)
    return f"[ {expr.op} {' '.join(render_expr(child) for child in expr.children)} ]"


def random_expr(
    rng: random.Random,
    *,
    depth: int,
    max_args: int,
    leaf_prob: float = 0.25,
) -> ListExpr:
    if depth <= 0 or rng.random() < leaf_prob:
        return ListExpr(None, value=rng.randrange(10))
    arity = rng.randint(2, max_args)
    return ListExpr(
        rng.choice(OPS),
        children=tuple(
            random_expr(rng, depth=depth - 1, max_args=max_args, leaf_prob=leaf_prob)
            for _ in range(arity)
        ),
    )


def render_listops_example(
    *,
    seed: int,
    graph_id: str,
    depth: int = 4,
    max_args: int = 4,
    positive: bool = True,
) -> ListOpsExample:
    rng = random.Random(seed)
    expr = random_expr(rng, depth=depth, max_args=max_args)
    value = eval_expr(expr)
    if positive:
        candidate = value
    else:
        choices = [digit for digit in range(10) if digit != value]
        candidate = rng.choice(choices)
    expression = render_expr(expr)
    prefix = f"Expression: {expression}. Does it evaluate to {candidate}? Answer"
    answer = "yes" if positive else "no"
    return ListOpsExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        correct=positive,
        expression=expression,
        value=value,
        candidate=candidate,
    )


class ListOpsDataset(Dataset):
    """Balanced ListOps-style yes/no evaluation examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        depth: int = 4,
        max_args: int = 4,
    ) -> None:
        self.examples = [
            render_listops_example(
                seed=seed + idx,
                graph_id=f"listops{idx}",
                depth=depth,
                max_args=max_args,
                positive=idx % 2 == 0,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ListOpsExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[ListOpsExample]) -> list[ListOpsExample]:
    return batch
