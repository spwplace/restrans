"""Verifiable lambda-calculus tasks for structural posttraining.

The tasks in this module are intentionally small and exact.  They are meant to
turn the original "semantic equivalence class of programs" idea into a binary
answer-token objective with a verifier:

* beta-normal-form equivalence: do two terms reduce to the same normal form?
* one-step beta validity: is the proposed successor the normal-order beta step?

Both tasks emit the same ``prefix`` / ``answer`` / ``full_text`` interface used
by the rest of the structural probe stack, so they can be used for SFT, DPO,
GRPO-style exact binary RL, and representation diagnostics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from .lambda_generator import Abs, App, Const, LambdaGenerator, T_INT, Term, Var, _beta_reduce_once, normal_form, typecheck
from .mutation_engine import BisimilarMutation, MutatedProgram


@dataclass(frozen=True)
class LambdaTaskExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    positive: bool
    left: str
    right: str
    task_kind: str
    metadata: dict[str, Any]


def _term_to_text(term: Term) -> str:
    return term.to_string().replace("λ", "lambda")


def _normal_form_text(term: Term, max_steps: int = 1_000) -> str:
    return _term_to_text(normal_form(term, max_steps=max_steps))


def _simple_redex() -> Term:
    return App(Abs(T_INT, Var(0)), Const(1))


def _identity_redex(term: Term) -> Term:
    try:
        ty = typecheck(term, [])
    except Exception:
        ty = T_INT
    return App(Abs(ty, Var(0)), term)


def _random_reducible_term(
    gen: LambdaGenerator,
    rng: random.Random,
    *,
    max_depth: int,
    max_size: int,
    max_steps: int,
) -> Term:
    """Construct a closed term with a guaranteed normal-order beta cascade."""
    try:
        base = _random_closed_term(
            gen,
            rng,
            max_depth=max(1, max_depth - 1),
            max_size=max(4, max_size // 2),
        )
    except Exception:
        base = Const(rng.randrange(3))
    term = base
    n_wrappers = rng.randint(1, max(1, max_steps))
    for _ in range(n_wrappers):
        term = _identity_redex(term)
    return term


def _random_closed_term(
    gen: LambdaGenerator,
    rng: random.Random,
    *,
    max_depth: int,
    max_size: int,
    target_type: object | None = None,
) -> Term:
    for _ in range(100):
        try:
            if target_type is not None:
                return gen.random_term(
                    max_depth=max_depth,
                    max_size=max_size,
                    target_type=target_type,
                    ctx=[],
                )
            return gen.random_term_diverse(
                max_depth=max_depth,
                max_size=max_size,
                strategy_weights={
                    "shallow_wide": rng.random(),
                    "deep_narrow": rng.random(),
                    "variable_rich": rng.random(),
                    "abstraction_rich": rng.random(),
                },
            )
        except Exception:
            continue
    return gen.random_term(max_depth=max(1, max_depth), max_size=max(4, max_size))


def _mutated_equivalent(
    term: Term,
    *,
    seed: int,
    n_mutations: int,
) -> tuple[Term, int]:
    engine = BisimilarMutation(seed=seed, verify=True)
    try:
        target_type = typecheck(term, [])
    except Exception:
        target_type = None
    program = MutatedProgram(
        origin=term,
        current=term,
        ctx=[],
        target_type=target_type,
    )
    mutated = engine.mutate(program, n_mutations=max(1, n_mutations))
    if mutated.verify_bisimilarity():
        return mutated.current, mutated.edit_count
    return normal_form(term), 0


def render_lambda_equivalence_example(
    *,
    seed: int,
    graph_id: str,
    positive: bool,
    max_depth: int = 4,
    max_size: int = 14,
    max_mutations: int = 8,
    hard_negative_same_type: bool = True,
) -> LambdaTaskExample:
    rng = random.Random(seed)
    gen = LambdaGenerator(seed=seed)
    left = _random_closed_term(gen, rng, max_depth=max_depth, max_size=max_size)
    left_nf = _normal_form_text(left)
    edit_count = 0

    if positive:
        if rng.random() < 0.65:
            n_mutations = rng.randint(1, max(1, max_mutations))
            right, edit_count = _mutated_equivalent(
                left,
                seed=seed + 123_457,
                n_mutations=n_mutations,
            )
        else:
            right = normal_form(left)
        equivalent = _normal_form_text(right) == left_nf
        if not equivalent:
            right = normal_form(left)
            edit_count = 0
            equivalent = True
    else:
        target_type = None
        if hard_negative_same_type:
            try:
                target_type = typecheck(left, [])
            except Exception:
                target_type = None
        right = left
        right_nf = left_nf
        for _ in range(200):
            candidate = _random_closed_term(
                gen,
                rng,
                max_depth=max_depth,
                max_size=max_size,
                target_type=target_type,
            )
            candidate_nf = _normal_form_text(candidate)
            if candidate_nf != left_nf:
                right = candidate
                right_nf = candidate_nf
                break
        equivalent = right_nf == left_nf
        if equivalent:
            # Fallback that is intentionally simple but still verified.
            right = _random_closed_term(gen, rng, max_depth=1, max_size=4)
            equivalent = _normal_form_text(right) == left_nf

    left_text = _term_to_text(left)
    right_text = _term_to_text(right)
    prefix = (
        f"Program A: {left_text}. Program B: {right_text}. "
        "Do these programs have the same beta normal form? Answer"
    )
    answer = "yes" if equivalent else "no"
    return LambdaTaskExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        positive=equivalent,
        left=left_text,
        right=right_text,
        task_kind="lambda_equivalence",
        metadata={
            "left_nf": left_nf,
            "right_nf": _normal_form_text(right),
            "edit_count": edit_count,
            "same_type_negative": hard_negative_same_type,
        },
    )


def render_lambda_beta_step_example(
    *,
    seed: int,
    graph_id: str,
    positive: bool,
    max_depth: int = 5,
    max_size: int = 18,
) -> LambdaTaskExample:
    rng = random.Random(seed)
    gen = LambdaGenerator(seed=seed)

    source = _random_reducible_term(
        gen,
        rng,
        max_depth=max_depth,
        max_size=max_size,
        max_steps=3,
    )
    reduced, actual_next = _beta_reduce_once(source)
    if not reduced:
        source = _simple_redex()
        reduced, actual_next = _beta_reduce_once(source)

    if positive and reduced:
        candidate = actual_next
        valid = True
        negative_kind = ""
    else:
        valid = False
        negative_kind = rng.choice(["normal_form", "unrelated", "mutated_equiv"])
        if negative_kind == "normal_form":
            candidate = normal_form(source)
            if candidate == actual_next:
                candidate = _random_closed_term(gen, rng, max_depth=max_depth, max_size=max_size)
        elif negative_kind == "mutated_equiv":
            candidate, _edits = _mutated_equivalent(
                source,
                seed=seed + 311,
                n_mutations=rng.randint(1, 4),
            )
            if candidate == actual_next:
                candidate = normal_form(source)
        else:
            candidate = _random_closed_term(gen, rng, max_depth=max_depth, max_size=max_size)
        valid = reduced and candidate == actual_next
        if valid:
            candidate = _random_closed_term(gen, rng, max_depth=1, max_size=4)
            valid = False

    left_text = _term_to_text(source)
    right_text = _term_to_text(candidate)
    prefix = (
        f"Current program: {left_text}. Proposed next program: {right_text}. "
        "Is the proposal the next normal-order beta-reduction step? Answer"
    )
    answer = "yes" if valid else "no"
    return LambdaTaskExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        positive=valid,
        left=left_text,
        right=right_text,
        task_kind="lambda_beta_step",
        metadata={
            "has_reduction": reduced,
            "actual_next": _term_to_text(actual_next),
            "negative_kind": negative_kind,
        },
    )


def beta_trace(term: Term, max_steps: int = 16) -> list[Term]:
    """Return a bounded normal-order beta trace starting at ``term``."""
    trace = [term]
    for _ in range(max_steps):
        reduced, next_term = _beta_reduce_once(trace[-1])
        if not reduced:
            break
        trace.append(next_term)
    return trace


def _render_trace(terms: list[Term]) -> str:
    return " ; ".join(f"step {idx}: {_term_to_text(term)}" for idx, term in enumerate(terms))


def _local_step_prompt(left: Term, right: Term) -> str:
    return (
        f"Current program: {_term_to_text(left)}. Proposed next program: {_term_to_text(right)}. "
        "Is the proposal the next normal-order beta-reduction step? Answer"
    )


def render_lambda_trace_example(
    *,
    seed: int,
    graph_id: str,
    positive: bool,
    max_depth: int = 5,
    max_size: int = 18,
    max_trace_steps: int = 8,
) -> LambdaTaskExample:
    """Ask whether a full proposed trace is a valid beta-reduction cascade."""
    rng = random.Random(seed)
    gen = LambdaGenerator(seed=seed)

    source = _random_reducible_term(
        gen,
        rng,
        max_depth=max_depth,
        max_size=max_size,
        max_steps=max_trace_steps,
    )
    trace = beta_trace(source, max_steps=max_trace_steps)
    if len(trace) < 2 or not _trace_is_exact_beta_cascade(trace):
        source = _simple_redex()
        trace = beta_trace(source, max_steps=max_trace_steps)

    candidate_trace = list(trace)
    valid = positive and _trace_is_exact_beta_cascade(trace)
    negative_kind = ""
    if not valid:
        negative_kind = rng.choice(["corrupt_step", "skip_step", "wrong_terminal", "swap_steps"])
        if negative_kind == "corrupt_step" and len(candidate_trace) >= 2:
            idx = rng.randrange(1, len(candidate_trace))
            candidate_trace[idx] = _random_closed_term(gen, rng, max_depth=max_depth, max_size=max_size)
        elif negative_kind == "skip_step" and len(candidate_trace) >= 3:
            del candidate_trace[rng.randrange(1, len(candidate_trace) - 1)]
        elif negative_kind == "swap_steps" and len(candidate_trace) >= 4:
            idx = rng.randrange(1, len(candidate_trace) - 2)
            candidate_trace[idx], candidate_trace[idx + 1] = candidate_trace[idx + 1], candidate_trace[idx]
        else:
            candidate_trace[-1] = _random_closed_term(gen, rng, max_depth=max_depth, max_size=max_size)
        valid = _trace_is_exact_beta_cascade(candidate_trace)
        if valid:
            candidate_trace[-1] = _random_closed_term(gen, rng, max_depth=1, max_size=4)
            valid = False

    source_text = _term_to_text(source)
    proposed_text = _render_trace(candidate_trace)
    prefix = (
        f"Start program: {source_text}. Proposed beta trace: {proposed_text}. "
        "Is this entire trace a valid normal-order beta-reduction cascade? Answer"
    )
    answer = "yes" if valid else "no"
    return LambdaTaskExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        positive=valid,
        left=source_text,
        right=proposed_text,
        task_kind="lambda_trace",
        metadata={
            "trace_len": len(candidate_trace),
            "gold_trace_len": len(trace),
            "negative_kind": negative_kind,
            "gold_terminal": _term_to_text(trace[-1]),
            "local_step_prompts": [
                _local_step_prompt(left, right)
                for left, right in zip(candidate_trace, candidate_trace[1:])
            ],
            **_trace_process_metadata(candidate_trace),
        },
    )


def _trace_is_exact_beta_cascade(trace: list[Term]) -> bool:
    if len(trace) < 2:
        return False
    for left, right in zip(trace, trace[1:]):
        reduced, actual = _beta_reduce_once(left)
        if not reduced or actual != right:
            return False
    reduced, _next = _beta_reduce_once(trace[-1])
    return not reduced


def _trace_process_metadata(trace: list[Term]) -> dict[str, int | float | bool]:
    """Record how much of a proposed cascade is locally correct.

    Whole-trace validity is the closure of the one-step beta rule plus a
    terminal normal-form check.  Keeping this denser verifier metadata lets us
    train one-step first, then ask whether that local competence composes.
    """
    if len(trace) < 2:
        return {
            "valid_local_steps": 0,
            "total_local_steps": 0,
            "terminal_normal": False,
            "process_score": 0.0,
        }
    valid_local_steps = 0
    for left, right in zip(trace, trace[1:]):
        reduced, actual = _beta_reduce_once(left)
        if reduced and actual == right:
            valid_local_steps += 1
        else:
            break
    reduced, _next = _beta_reduce_once(trace[-1])
    terminal_normal = not reduced
    total_local_steps = len(trace) - 1
    process_score = (valid_local_steps + int(terminal_normal)) / max(total_local_steps + 1, 1)
    return {
        "valid_local_steps": valid_local_steps,
        "total_local_steps": total_local_steps,
        "terminal_normal": terminal_normal,
        "process_score": process_score,
    }


class LambdaEquivalenceDataset(Dataset):
    """Balanced verified beta-normal-form equivalence examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        depth: int = 4,
        max_size: int = 14,
        max_mutations: int = 8,
    ) -> None:
        self.examples = [
            render_lambda_equivalence_example(
                seed=seed + idx,
                graph_id=f"lambdaeq{idx}",
                positive=idx % 2 == 0,
                max_depth=depth,
                max_size=max_size,
                max_mutations=max_mutations,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> LambdaTaskExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


class LambdaBetaStepDataset(Dataset):
    """Balanced verified one-step beta-reduction examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        depth: int = 5,
        max_size: int = 18,
    ) -> None:
        self.examples = [
            render_lambda_beta_step_example(
                seed=seed + idx,
                graph_id=f"lambdastep{idx}",
                positive=idx % 2 == 0,
                max_depth=depth,
                max_size=max_size,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> LambdaTaskExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


class LambdaTraceDataset(Dataset):
    """Balanced whole-trace beta-reduction validation examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        depth: int = 5,
        max_size: int = 18,
        max_trace_steps: int = 8,
    ) -> None:
        self.examples = [
            render_lambda_trace_example(
                seed=seed + idx,
                graph_id=f"lambdatrace{idx}",
                positive=idx % 2 == 0,
                max_depth=depth,
                max_size=max_size,
                max_trace_steps=max_trace_steps,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> LambdaTaskExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[LambdaTaskExample]) -> list[LambdaTaskExample]:
    return batch
