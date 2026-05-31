#!/usr/bin/env python3
"""Train causal transformers on exact sequence-prediction verifier tasks.

The yes/no verifier harness is useful, but it does not directly test whether a
model can *produce* the next formal state.  This script keeps the same model
family and tokenizer, but supervises every token of a target sequence:

* lambda_beta_next: current lambda term -> next normal-order beta step
* lambda_normal_form: lambda term -> beta normal form
* vm_step_next: current stack + instruction -> next stack

Evaluation reports target-token accuracy and exact-match accuracy under teacher
forcing.  This is intentionally modest: it is a reproducible sequence task with
an exact generator/verifier, suitable for architecture comparisons before we
claim anything about broader language modeling.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from resonance.device import enable_deterministic, get_device, set_seed  # noqa: E402
from story_topology_eval import build_model, jsonify  # noqa: E402
from synthetic.lambda_generator import Const, LambdaGenerator, T_INT, Term, Var, Abs, App, _beta_reduce_once, normal_form, typecheck  # noqa: E402
from synthetic.lambda_tasks import _normal_form_text, _term_to_text  # noqa: E402
from synthetic.semantic_story import StoryTokenizer  # noqa: E402
from synthetic.vm_tasks import execute, render_op, render_stack, step  # noqa: E402


TASKS = ("lambda_beta_next", "lambda_normal_form", "vm_step_next")


@dataclass(frozen=True)
class SequenceExample:
    prefix: str
    target: str
    full_text: str
    graph_id: str
    task_kind: str
    metadata: dict[str, Any]


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
                return gen.random_term(max_depth=max_depth, max_size=max_size, target_type=target_type, ctx=[])
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
    return Const(rng.randrange(5))


def _identity_redex(term: Term) -> Term:
    try:
        ty = typecheck(term, [])
    except Exception:
        ty = T_INT
    return App(Abs(ty, Var(0)), term)


def random_reducible_term(seed: int, *, depth: int, max_size: int, wrappers: int) -> Term:
    rng = random.Random(seed)
    gen = LambdaGenerator(seed=seed)
    base = _random_closed_term(
        gen,
        rng,
        max_depth=max(1, depth - 1),
        max_size=max(4, max_size // 2),
    )
    term = base
    for _ in range(max(1, wrappers)):
        term = _identity_redex(term)
    return term


def render_lambda_beta_next(seed: int, graph_id: str, *, depth: int, max_size: int, max_trace_steps: int) -> SequenceExample:
    rng = random.Random(seed)
    source = random_reducible_term(
        seed,
        depth=depth,
        max_size=max_size,
        wrappers=rng.randint(1, max(1, max_trace_steps)),
    )
    reduced, target = _beta_reduce_once(source)
    if not reduced:
        source = App(Abs(T_INT, Var(0)), Const(1))
        _reduced, target = _beta_reduce_once(source)
    prefix = f"Current lambda program: {_term_to_text(source)}. Next normal-order beta step:"
    target_text = _term_to_text(target)
    return SequenceExample(
        prefix=prefix,
        target=target_text,
        full_text=f"{prefix} {target_text}.",
        graph_id=graph_id,
        task_kind="lambda_beta_next",
        metadata={"source": _term_to_text(source), "target": target_text},
    )


def render_lambda_normal_form(seed: int, graph_id: str, *, depth: int, max_size: int, max_trace_steps: int) -> SequenceExample:
    rng = random.Random(seed)
    source = random_reducible_term(
        seed,
        depth=depth,
        max_size=max_size,
        wrappers=rng.randint(1, max(1, max_trace_steps)),
    )
    target_text = _normal_form_text(source, max_steps=1_000)
    prefix = f"Lambda program: {_term_to_text(source)}. Beta normal form:"
    return SequenceExample(
        prefix=prefix,
        target=target_text,
        full_text=f"{prefix} {target_text}.",
        graph_id=graph_id,
        task_kind="lambda_normal_form",
        metadata={"source": _term_to_text(source), "target": target_text},
    )


def random_valid_vm_step(seed: int, *, length: int, modulus: int) -> tuple[tuple[int, ...], tuple[str, int | None], tuple[int, ...]]:
    rng = random.Random(seed)
    program = []
    stack: tuple[int, ...] = ()
    for _ in range(max(1, length)):
        candidates: list[tuple[str, int | None]] = [("PUSH", rng.randrange(modulus))]
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
    ok, trace = execute(program, modulus=modulus)
    if not ok or not program:
        return (), ("PUSH", 0), (0,)
    idx = rng.randrange(len(program))
    return trace[idx], program[idx], trace[idx + 1]


def render_vm_step_next(seed: int, graph_id: str, *, depth: int, modulus: int) -> SequenceExample:
    stack, op, next_stack = random_valid_vm_step(seed, length=depth, modulus=modulus)
    prefix = (
        f"Stack VM modulus: {modulus}. Current stack: {render_stack(stack)}. "
        f"Instruction: {render_op(op)}. Next stack:"
    )
    target = render_stack(next_stack)
    return SequenceExample(
        prefix=prefix,
        target=target,
        full_text=f"{prefix} {target}.",
        graph_id=graph_id,
        task_kind="vm_step_next",
        metadata={"stack": render_stack(stack), "op": render_op(op), "target": target},
    )


class SequencePredictionDataset(Dataset):
    def __init__(
        self,
        *,
        task: str,
        n_examples: int,
        seed: int,
        depth: int,
        max_size: int,
        max_trace_steps: int,
        modulus: int,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task: {task}")
        examples = []
        for idx in range(n_examples):
            kwargs = {"seed": seed + idx, "graph_id": f"{task}{idx}"}
            if task == "lambda_beta_next":
                examples.append(
                    render_lambda_beta_next(
                        **kwargs,
                        depth=depth,
                        max_size=max_size,
                        max_trace_steps=max_trace_steps,
                    )
                )
            elif task == "lambda_normal_form":
                examples.append(
                    render_lambda_normal_form(
                        **kwargs,
                        depth=depth,
                        max_size=max_size,
                        max_trace_steps=max_trace_steps,
                    )
                )
            else:
                examples.append(render_vm_step_next(**kwargs, depth=depth, modulus=modulus))
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SequenceExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[SequenceExample]) -> list[SequenceExample]:
    return batch


def encode_batch(
    tokenizer: StoryTokenizer,
    batch: list[SequenceExample],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = [example.full_text for example in batch]
    input_ids = tokenizer.batch_encode(texts, max_length=max_length, device=device)
    labels = torch.full_like(input_ids, -100)
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, example in enumerate(batch):
        prefix_len = 1 + len(tokenizer._words(example.prefix))
        target_words = tokenizer._words(example.target)
        if prefix_len + len(target_words) >= max_length:
            raise ValueError(
                f"max_length={max_length} too short for target sequence; "
                f"need at least {prefix_len + len(target_words) + 1}"
            )
        for offset, word in enumerate(target_words):
            # Causal alignment: logits at the position before a target token
            # must predict that target token. The target token itself is still
            # present in the input as teacher-forced context for later tokens.
            pred_pos = prefix_len - 1 + offset
            labels[row, pred_pos] = tokenizer.word2id.get(word, tokenizer.unk_id)
            target_mask[row, pred_pos] = True
    return input_ids, labels, target_mask


def target_loss(
    model: torch.nn.Module,
    tokenizer: StoryTokenizer,
    batch: list[SequenceExample],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, labels, target_mask = encode_batch(
        tokenizer,
        batch,
        max_length=max_length,
        device=device,
    )
    out = model(input_ids, labels=None, return_hidden=False)
    logits = out["logits"]
    loss = F.cross_entropy(
        logits[target_mask],
        labels[target_mask],
    )
    return loss, logits, labels, target_mask


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: SequencePredictionDataset,
    tokenizer: StoryTokenizer,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    losses = []
    total_tokens = 0
    correct_tokens = 0
    exact = 0
    total_examples = 0
    for batch in loader:
        loss, logits, labels, target_mask = target_loss(
            model,
            tokenizer,
            batch,
            max_length=max_length,
            device=device,
        )
        losses.append(float(loss.detach().item()))
        pred = logits.argmax(dim=-1)
        target_positions = target_mask.nonzero(as_tuple=False)
        total_tokens += int(target_positions.size(0))
        correct_tokens += int((pred[target_mask] == labels[target_mask]).sum().item())
        for row in range(labels.size(0)):
            mask = target_mask[row]
            exact += int(torch.equal(pred[row][mask], labels[row][mask]))
            total_examples += 1
    token_acc = correct_tokens / max(total_tokens, 1)
    exact_acc = exact / max(total_examples, 1)
    return {
        "target_loss": statistics.fmean(losses) if losses else 0.0,
        "target_ppl": math.exp(min(20.0, statistics.fmean(losses))) if losses else 1.0,
        "token_acc": token_acc,
        "exact_acc": exact_acc,
    }


def train_condition(
    args: argparse.Namespace,
    condition: str,
    seed: int,
    train_ds: SequencePredictionDataset,
    val_ds: SequencePredictionDataset,
    tokenizer: StoryTokenizer,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device(args.device)
    model, config = build_model(args, condition, tokenizer)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_examples)
    if args.skip_before_eval:
        before = {
            "target_loss": 0.0,
            "target_ppl": 1.0,
            "token_acc": 0.0,
            "exact_acc": 0.0,
        }
    else:
        before = evaluate(model, val_ds, tokenizer, batch_size=args.batch_size, max_length=args.max_length, device=device)
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            loss, _logits, _labels, _mask = target_loss(
                model,
                tokenizer,
                batch,
                max_length=args.max_length,
                device=device,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        row = {"epoch": epoch + 1, "train_loss": statistics.fmean(losses)}
        if args.eval_each_epoch:
            row.update({f"val_{k}": v for k, v in evaluate(model, val_ds, tokenizer, batch_size=args.batch_size, max_length=args.max_length, device=device).items()})
        history.append(row)
        msg = f"[{condition} seed={seed}] epoch={epoch + 1} train_loss={row['train_loss']:.4f}"
        if args.eval_each_epoch:
            msg += f" val_exact={row['val_exact_acc']:.4f} val_token={row['val_token_acc']:.4f}"
        print(msg, flush=True)
    after = evaluate(model, val_ds, tokenizer, batch_size=args.batch_size, max_length=args.max_length, device=device)
    return {
        "task": args.task,
        "condition": condition,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "config": jsonify(getattr(config, "__dict__", {})),
        "before": before,
        "after": after,
        "history": history,
        "elapsed_sec": time.perf_counter() - start,
    }


def summarise(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["condition"], []).append(run)
    out = {}
    for condition, rows in grouped.items():
        exact = [float(row["after"]["exact_acc"]) for row in rows]
        token = [float(row["after"]["token_acc"]) for row in rows]
        loss = [float(row["after"]["target_loss"]) for row in rows]
        before_exact = [float(row["before"]["exact_acc"]) for row in rows]
        out[condition] = {
            "n": float(len(rows)),
            "params": float(rows[0]["params"]),
            "before_exact_mean": statistics.fmean(before_exact),
            "exact_acc_mean": statistics.fmean(exact),
            "exact_acc_std": statistics.stdev(exact) if len(exact) > 1 else 0.0,
            "exact_gain": statistics.fmean(exact) - statistics.fmean(before_exact),
            "token_acc_mean": statistics.fmean(token),
            "target_loss_mean": statistics.fmean(loss),
        }
    return out


def write_report(path: Path, args: argparse.Namespace, runs: list[dict[str, Any]]) -> None:
    summary = summarise(runs)
    lines = [
        "# Sequence Prediction",
        "",
        f"Task: `{args.task}`.",
        f"Train examples: `{args.train_examples}`, validation examples: `{args.val_examples}`, epochs: `{args.epochs}`.",
        "",
        "| Condition | Params | Before exact | Exact acc | Gain | Token acc | Target loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in summary.items():
        lines.append(
            f"| {condition} | {int(row['params']):,} | {row['before_exact_mean']:.4f} | "
            f"{row['exact_acc_mean']:.4f} | {row['exact_gain']:+.4f} | "
            f"{row['token_acc_mean']:.4f} | {row['target_loss_mean']:.4f} |"
        )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="lambda_beta_next")
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/sequence_prediction"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", default="standard_alibi,phase_dynamic_qk_film_alibi_normalized")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_examples", type=int, default=2048)
    parser.add_argument("--val_examples", type=int, default=768)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--embed_dim", type=int, default=160)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--phase_init_std", type=float, default=0.3)
    parser.add_argument("--story_phase_prior_std", type=float, default=1.2)
    parser.add_argument("--story_phase_prior_noise", type=float, default=0.05)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--dataset_seed", type=int, default=9600)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--val_depth", type=int, default=None)
    parser.add_argument("--max_size", type=int, default=18)
    parser.add_argument("--max_trace_steps", type=int, default=8)
    parser.add_argument("--modulus", type=int, default=17)
    parser.add_argument("--eval_each_epoch", action="store_true")
    parser.add_argument("--skip_before_eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_deterministic(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_depth = args.depth if args.val_depth is None else args.val_depth
    train_ds = SequencePredictionDataset(
        task=args.task,
        n_examples=args.train_examples,
        seed=args.dataset_seed,
        depth=args.depth,
        max_size=args.max_size,
        max_trace_steps=args.max_trace_steps,
        modulus=args.modulus,
    )
    val_ds = SequencePredictionDataset(
        task=args.task,
        n_examples=args.val_examples,
        seed=args.dataset_seed + 100_000,
        depth=val_depth,
        max_size=args.max_size,
        max_trace_steps=args.max_trace_steps,
        modulus=args.modulus,
    )
    tokenizer = StoryTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(train_ds.all_texts() + val_ds.all_texts())
    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    runs = []
    for seed in args.seeds:
        for condition in conditions:
            runs.append(train_condition(args, condition, seed, train_ds, val_ds, tokenizer))
    results = {
        "task": args.task,
        "summary": summarise(runs),
        "runs": runs,
        "config": vars(args),
    }
    (args.output_dir / "results.json").write_text(json.dumps(jsonify(results), indent=2))
    write_report(args.output_dir / "report.md", args, runs)
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
