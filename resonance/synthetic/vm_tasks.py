"""Small verified stack-VM tasks for structural training.

This gives us an executable non-lambda substrate with exact verifiers.  The
model sees programs, traces, and final states as text; labels come from an
interpreter, not from templates.  These tasks are useful for RLVR because each
sample has both an outcome reward and step-level process checks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from torch.utils.data import Dataset


@dataclass(frozen=True)
class VMExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    positive: bool
    left: str
    right: str
    task_kind: str
    metadata: dict[str, Any]


Op = tuple[str, int | None]
Stack = tuple[int, ...]


def render_op(op: Op) -> str:
    name, arg = op
    return f"PUSH {arg}" if name == "PUSH" else name


def render_program(program: Iterable[Op]) -> str:
    return " ; ".join(render_op(op) for op in program)


def render_stack(stack: Stack) -> str:
    return "[" + " ".join(str(value) for value in stack) + "]"


def step(stack: Stack, op: Op, *, modulus: int = 17) -> Stack | None:
    values = list(stack)
    name, arg = op
    if name == "PUSH":
        assert arg is not None
        values.append(int(arg) % modulus)
    elif name == "ADD":
        if len(values) < 2:
            return None
        b = values.pop()
        a = values.pop()
        values.append((a + b) % modulus)
    elif name == "SUB":
        if len(values) < 2:
            return None
        b = values.pop()
        a = values.pop()
        values.append((a - b) % modulus)
    elif name == "MUL":
        if len(values) < 2:
            return None
        b = values.pop()
        a = values.pop()
        values.append((a * b) % modulus)
    elif name == "DUP":
        if not values:
            return None
        values.append(values[-1])
    elif name == "SWAP":
        if len(values) < 2:
            return None
        values[-1], values[-2] = values[-2], values[-1]
    elif name == "POP":
        if not values:
            return None
        values.pop()
    else:
        return None
    return tuple(values)


def execute(program: list[Op], *, modulus: int = 17) -> tuple[bool, list[Stack]]:
    trace: list[Stack] = [()]
    stack: Stack = ()
    for op in program:
        next_stack = step(stack, op, modulus=modulus)
        if next_stack is None:
            return False, trace
        stack = next_stack
        trace.append(stack)
    return True, trace


def _random_valid_program(rng: random.Random, *, length: int, modulus: int) -> list[Op]:
    program: list[Op] = []
    stack: Stack = ()
    for _ in range(length):
        candidates: list[Op] = [("PUSH", rng.randrange(modulus))]
        if stack:
            candidates.extend([("DUP", None), ("POP", None)])
        if len(stack) >= 2:
            candidates.extend([("ADD", None), ("SUB", None), ("MUL", None), ("SWAP", None)])
        rng.shuffle(candidates)
        for op in candidates:
            next_stack = step(stack, op, modulus=modulus)
            if next_stack is not None:
                program.append(op)
                stack = next_stack
                break
    return program


def _random_program(rng: random.Random, *, length: int, modulus: int) -> list[Op]:
    ops = ["ADD", "SUB", "MUL", "DUP", "SWAP", "POP"]
    program: list[Op] = []
    for _ in range(length):
        if rng.random() < 0.45:
            program.append(("PUSH", rng.randrange(modulus)))
        else:
            program.append((rng.choice(ops), None))
    return program


def _trace_is_valid(program: list[Op], trace: list[Stack], *, modulus: int) -> bool:
    ok, actual = execute(program, modulus=modulus)
    return ok and actual == trace


def _trace_process_metadata(program: list[Op], trace: list[Stack], *, modulus: int) -> dict[str, int | float | bool]:
    if not trace:
        return {
            "valid_local_steps": 0,
            "total_local_steps": len(program),
            "terminal_matches": False,
            "process_score": 0.0,
        }
    valid_local_steps = 0
    stack = trace[0]
    for op, proposed in zip(program, trace[1:]):
        actual = step(stack, op, modulus=modulus)
        if actual is None or actual != proposed:
            break
        valid_local_steps += 1
        stack = proposed
    ok, actual_trace = execute(program, modulus=modulus)
    terminal_matches = ok and len(trace) == len(actual_trace) and trace[-1] == actual_trace[-1]
    total_local_steps = len(program)
    process_score = (valid_local_steps + int(terminal_matches)) / max(total_local_steps + 1, 1)
    return {
        "valid_local_steps": valid_local_steps,
        "total_local_steps": total_local_steps,
        "terminal_matches": terminal_matches,
        "process_score": process_score,
    }


def render_trace(trace: list[Stack]) -> str:
    return " -> ".join(render_stack(stack) for stack in trace)


def _local_step_prompt(stack: Stack, op: Op, next_stack: Stack, *, modulus: int) -> str:
    return (
        f"Stack VM modulus: {modulus}. Current stack: {render_stack(stack)}. "
        f"Instruction: {render_op(op)}. Proposed next stack: {render_stack(next_stack)}. "
        "Is this one VM step valid? Answer"
    )


def render_vm_step_example(
    *,
    seed: int,
    graph_id: str,
    positive: bool,
    program_length: int = 8,
    modulus: int = 17,
) -> VMExample:
    rng = random.Random(seed)
    program = _random_valid_program(rng, length=program_length, modulus=modulus)
    _ok, trace = execute(program, modulus=modulus)
    step_idx = rng.randrange(len(program))
    current_stack = trace[step_idx]
    op = program[step_idx]
    actual_next = trace[step_idx + 1]

    if positive:
        candidate = actual_next
        valid = True
        negative_kind = ""
    else:
        negative_kind = rng.choice(["wrong_value", "wrong_depth", "previous_stack", "random_stack"])
        if negative_kind == "previous_stack":
            candidate = current_stack
        elif negative_kind == "wrong_depth":
            candidate = tuple(rng.randrange(modulus) for _ in range(max(0, len(actual_next) + rng.choice([-1, 1]))))
        elif negative_kind == "wrong_value" and actual_next:
            values = list(actual_next)
            idx = rng.randrange(len(values))
            values[idx] = (values[idx] + rng.randrange(1, modulus)) % modulus
            candidate = tuple(values)
        else:
            candidate = tuple(rng.randrange(modulus) for _ in range(rng.randint(0, 4)))
        valid = step(current_stack, op, modulus=modulus) == candidate
        if valid:
            candidate = ((candidate[0] + 1) % modulus,) if candidate else (1,)
            valid = False

    prefix = _local_step_prompt(current_stack, op, candidate, modulus=modulus)
    answer = "yes" if valid else "no"
    return VMExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        positive=valid,
        left=f"{render_stack(current_stack)} ; {render_op(op)}",
        right=render_stack(candidate),
        task_kind="vm_step",
        metadata={
            "program_length": len(program),
            "step_idx": step_idx,
            "actual_next": render_stack(actual_next),
            "negative_kind": negative_kind,
        },
    )


def render_vm_trace_example(
    *,
    seed: int,
    graph_id: str,
    positive: bool,
    program_length: int = 8,
    modulus: int = 17,
) -> VMExample:
    rng = random.Random(seed)
    program = _random_valid_program(rng, length=program_length, modulus=modulus)
    _ok, trace = execute(program, modulus=modulus)
    candidate_trace = list(trace)
    negative_kind = ""
    valid = positive
    if not positive:
        negative_kind = rng.choice(["corrupt_state", "skip_state", "wrong_terminal", "swap_states"])
        if negative_kind == "corrupt_state" and len(candidate_trace) > 1:
            idx = rng.randrange(1, len(candidate_trace))
            candidate_trace[idx] = tuple(rng.randrange(modulus) for _ in range(rng.randint(0, 4)))
        elif negative_kind == "skip_state" and len(candidate_trace) > 2:
            del candidate_trace[rng.randrange(1, len(candidate_trace))]
        elif negative_kind == "swap_states" and len(candidate_trace) > 3:
            idx = rng.randrange(1, len(candidate_trace) - 1)
            candidate_trace[idx], candidate_trace[idx + 1] = candidate_trace[idx + 1], candidate_trace[idx]
        else:
            candidate_trace[-1] = tuple(rng.randrange(modulus) for _ in range(rng.randint(0, 4)))
        valid = _trace_is_valid(program, candidate_trace, modulus=modulus)
        if valid:
            candidate_trace[-1] = ((candidate_trace[-1][0] + 1) % modulus,) if candidate_trace[-1] else (1,)
            valid = False

    program_text = render_program(program)
    trace_text = render_trace(candidate_trace)
    prefix = (
        f"Stack VM modulus: {modulus}. Program: {program_text}. "
        f"Proposed stack trace: {trace_text}. Is this trace valid? Answer"
    )
    answer = "yes" if valid else "no"
    return VMExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        positive=valid,
        left=program_text,
        right=trace_text,
        task_kind="vm_trace",
        metadata={
            "program_length": len(program),
            "trace_len": len(candidate_trace),
            "negative_kind": negative_kind,
            "final_stack": render_stack(trace[-1]),
            "local_step_prompts": [
                _local_step_prompt(stack, op, next_stack, modulus=modulus)
                for stack, op, next_stack in zip(candidate_trace, program, candidate_trace[1:])
            ],
            **_trace_process_metadata(program, candidate_trace, modulus=modulus),
        },
    )


def render_vm_equivalence_example(
    *,
    seed: int,
    graph_id: str,
    positive: bool,
    program_length: int = 8,
    modulus: int = 17,
) -> VMExample:
    rng = random.Random(seed)
    left = _random_valid_program(rng, length=program_length, modulus=modulus)
    _left_ok, left_trace = execute(left, modulus=modulus)
    left_final = left_trace[-1]

    if positive:
        # The identity suffix perturbs the trace while preserving the final stack.
        right = list(left)
        if left_final:
            right.extend([("DUP", None), ("POP", None)])
        else:
            right.extend([("PUSH", 0), ("POP", None)])
        equivalent = execute(right, modulus=modulus)[1][-1] == left_final
    else:
        right = _random_program(rng, length=program_length, modulus=modulus)
        equivalent = False
        for _ in range(200):
            ok, right_trace = execute(right, modulus=modulus)
            if ok and right_trace[-1] != left_final:
                equivalent = False
                break
            right = _random_program(rng, length=program_length, modulus=modulus)
        else:
            right = list(left) + [("PUSH", 1)]
            equivalent = False

    left_text = render_program(left)
    right_text = render_program(right)
    prefix = (
        f"Stack VM modulus: {modulus}. Program A: {left_text}. Program B: {right_text}. "
        "Do these programs finish with the same stack? Answer"
    )
    answer = "yes" if equivalent else "no"
    return VMExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        positive=equivalent,
        left=left_text,
        right=right_text,
        task_kind="vm_equivalence",
        metadata={
            "program_length": len(left),
            "left_final": render_stack(left_final),
            "right_final": render_stack(execute(right, modulus=modulus)[1][-1]),
        },
    )


class VMTraceDataset(Dataset):
    """Balanced stack-VM trace-validation examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        depth: int = 8,
        modulus: int = 17,
    ) -> None:
        self.examples = [
            render_vm_trace_example(
                seed=seed + idx,
                graph_id=f"vmtrace{idx}",
                positive=idx % 2 == 0,
                program_length=depth,
                modulus=modulus,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> VMExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


class VMStepDataset(Dataset):
    """Balanced one-step stack-VM transition validation examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        depth: int = 8,
        modulus: int = 17,
    ) -> None:
        self.examples = [
            render_vm_step_example(
                seed=seed + idx,
                graph_id=f"vmstep{idx}",
                positive=idx % 2 == 0,
                program_length=depth,
                modulus=modulus,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> VMExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


class VMEquivalenceDataset(Dataset):
    """Balanced stack-VM final-state equivalence examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        depth: int = 8,
        modulus: int = 17,
    ) -> None:
        self.examples = [
            render_vm_equivalence_example(
                seed=seed + idx,
                graph_id=f"vmeq{idx}",
                positive=idx % 2 == 0,
                program_length=depth,
                modulus=modulus,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> VMExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[VMExample]) -> list[VMExample]:
    return batch
