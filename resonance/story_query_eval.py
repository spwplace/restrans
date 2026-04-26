#!/usr/bin/env python3
"""Temporal-reasoning evaluation over controlled natural-language stories.

The retrieval benchmark is useful plumbing, but it can saturate through lexical
overlap.  This harness asks yes/no questions whose answer is computed from the
transitive closure of a rendered event graph:

    fact: event A happened before event B
    fact: event B happened before event C
    query: did A happen before C?

The answer token is supervised directly through the causal language-model head.
This keeps the task close to next-token training while making the structural
target explicit and reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from resonance.device import enable_deterministic, get_device, set_seed
from resonance.interpretability import phase_intervention, resonance_weight_scale
from resonance.models import ResonanceTransformer
from story_topology_eval import build_model, jsonify
from synthetic.semantic_story import (
    NAMES,
    OBJECTS,
    PLACES,
    PREDICATES,
    StoryTokenizer,
    event_sentence,
    Event,
)


@dataclass(frozen=True)
class TemporalExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    query: tuple[int, int]
    reachable: bool


def _make_events(rng: random.Random, n_events: int) -> list[Event]:
    people = rng.sample(NAMES, min(3, len(NAMES)))
    objects = rng.sample(OBJECTS, min(n_events, len(OBJECTS)))
    place = rng.choice(PLACES)
    events = []
    for idx in range(n_events):
        predicate = rng.choice(PREDICATES)
        subject = people[idx % len(people)]
        obj = objects[idx % len(objects)]
        target = people[(idx + 1) % len(people)] if predicate == "gives" else None
        events.append(
            Event(
                event_id=f"e{idx + 1}",
                predicate=predicate,
                subject=subject,
                obj=obj,
                target=target,
                location=place,
            )
        )
    return events


def _make_edges(rng: random.Random, n_events: int, edge_prob: float) -> list[tuple[int, int]]:
    edges = [(idx, idx + 1) for idx in range(n_events - 1)]
    for i in range(n_events):
        for j in range(i + 2, n_events):
            if rng.random() < edge_prob:
                edges.append((i, j))
    rng.shuffle(edges)
    return edges


def _closure(n_events: int, edges: list[tuple[int, int]]) -> list[list[bool]]:
    reach = [[False for _ in range(n_events)] for _ in range(n_events)]
    for source, target in edges:
        reach[source][target] = True
    for k in range(n_events):
        for i in range(n_events):
            if reach[i][k]:
                for j in range(n_events):
                    reach[i][j] = reach[i][j] or reach[k][j]
    return reach


def _event_ref(event: Event, style: int) -> str:
    sentence = event_sentence(event, style="compact")
    if style % 2 == 0:
        return sentence[0].lower() + sentence[1:]
    return f"the moment when {sentence[0].lower()}{sentence[1:]}"


def render_temporal_example(
    *,
    seed: int,
    graph_id: str,
    n_events: int,
    edge_prob: float,
    variant: int,
    positive: bool,
) -> TemporalExample:
    rng = random.Random(seed)
    events = _make_events(rng, n_events)
    edges = _make_edges(rng, n_events, edge_prob)
    reach = _closure(n_events, edges)

    reachable = [(i, j) for i in range(n_events) for j in range(n_events) if reach[i][j]]
    unreachable = [
        (i, j)
        for i in range(n_events)
        for j in range(n_events)
        if i != j and not reach[i][j]
    ]
    if positive and reachable:
        query = rng.choice(reachable)
    else:
        query = rng.choice(unreachable)
    answer = "yes" if reach[query[0]][query[1]] else "no"

    event_lines = []
    for idx, event in enumerate(events):
        label = chr(ord("A") + idx)
        event_lines.append(f"Event {label}: {event_sentence(event)}.")
    edge_lines = []
    edge_order = list(edges)
    rng.shuffle(edge_order)
    for source, target in edge_order:
        left = chr(ord("A") + source)
        right = chr(ord("A") + target)
        if variant % 3 == 0:
            edge_lines.append(f"Event {left} happened before event {right}.")
        elif variant % 3 == 1:
            edge_lines.append(f"Before event {right}, event {left} happened.")
        else:
            edge_lines.append(f"The {left} event was earlier than the {right} event.")

    q_source, q_target = query
    query_text = (
        f"Question: Did {_event_ref(events[q_source], variant)} happen before "
        f"{_event_ref(events[q_target], variant + 1)}? Answer"
    )
    prefix = " ".join(event_lines + edge_lines + [query_text])
    full_text = f"{prefix} {answer}."
    return TemporalExample(
        prefix=prefix,
        answer=answer,
        full_text=full_text,
        graph_id=graph_id,
        query=query,
        reachable=answer == "yes",
    )


class TemporalQueryDataset(Dataset):
    def __init__(
        self,
        *,
        n_examples: int,
        seed: int,
        n_events: int = 5,
        edge_prob: float = 0.25,
    ) -> None:
        self.examples = [
            render_temporal_example(
                seed=seed + idx,
                graph_id=f"tg{idx}",
                n_events=n_events,
                edge_prob=edge_prob,
                variant=idx,
                positive=idx % 2 == 0,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TemporalExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[TemporalExample]) -> list[TemporalExample]:
    return batch


def encode_supervised_batch(
    tokenizer: StoryTokenizer,
    batch: list[TemporalExample],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = tokenizer.batch_encode([example.full_text for example in batch], max_length, device)
    labels = torch.full_like(input_ids, -100)
    answer_positions = []
    for row, example in enumerate(batch):
        prefix_len = 1 + len(tokenizer._words(example.prefix))
        if prefix_len >= max_length:
            raise ValueError("max_length is too short for temporal query examples")
        labels[row, prefix_len] = tokenizer.word2id[example.answer]
        answer_positions.append(prefix_len - 1)
    return input_ids, labels, torch.tensor(answer_positions, dtype=torch.long, device=device)


def answer_loss_and_logits(
    model: torch.nn.Module,
    tokenizer: StoryTokenizer,
    batch: list[TemporalExample],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, labels, answer_positions = encode_supervised_batch(
        tokenizer, batch, max_length=max_length, device=device
    )
    out = model(input_ids, labels=None)
    logits = out["logits"]
    yes_id = tokenizer.word2id["yes"]
    no_id = tokenizer.word2id["no"]
    row_ids = torch.arange(input_ids.size(0), device=device)
    answer_logits = logits[row_ids, answer_positions][:, [yes_id, no_id]]
    targets = torch.tensor(
        [0 if example.answer == "yes" else 1 for example in batch],
        dtype=torch.long,
        device=device,
    )
    loss = F.cross_entropy(answer_logits, targets)
    return loss, answer_logits, targets


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: TemporalQueryDataset,
    tokenizer: StoryTokenizer,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    total = 0
    correct = 0
    losses = []
    margins = []
    for batch in loader:
        loss, answer_logits, targets = answer_loss_and_logits(
            model, tokenizer, batch, max_length=max_length, device=device
        )
        predictions = answer_logits.argmax(dim=-1)
        total += int(targets.numel())
        correct += int((predictions == targets).sum().item())
        losses.append(float(loss.item()))
        signed_margin = answer_logits[:, 0] - answer_logits[:, 1]
        signed_margin = torch.where(targets == 0, signed_margin, -signed_margin)
        margins.extend(signed_margin.detach().cpu().tolist())
    return {
        "answer_acc": correct / max(total, 1),
        "answer_loss": statistics.fmean(losses) if losses else 0.0,
        "answer_margin": statistics.fmean(margins) if margins else 0.0,
    }


def evaluate_interventions(
    model: torch.nn.Module,
    dataset: TemporalQueryDataset,
    tokenizer: StoryTokenizer,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    args: argparse.Namespace,
    base: dict[str, float],
) -> dict[str, dict[str, float]]:
    if not isinstance(model, ResonanceTransformer):
        return {}
    values = {}
    for mode in args.phase_intervention_modes:
        with phase_intervention(model, mode=mode, sigma=args.phase_noise_sigma, seed=args.dataset_seed):
            metric = evaluate(
                model,
                dataset,
                tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            )
        values[f"phase_{mode}"] = {
            **metric,
            "answer_acc_delta": metric["answer_acc"] - base["answer_acc"],
            "answer_margin_delta": metric["answer_margin"] - base["answer_margin"],
        }
    for scale in args.resonance_scales:
        with resonance_weight_scale(model, scale):
            metric = evaluate(
                model,
                dataset,
                tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            )
        values[f"resonance_scale_{scale:g}"] = {
            **metric,
            "answer_acc_delta": metric["answer_acc"] - base["answer_acc"],
            "answer_margin_delta": metric["answer_margin"] - base["answer_margin"],
        }
    return values


def train_condition(
    args: argparse.Namespace,
    condition: str,
    seed: int,
    train_ds: TemporalQueryDataset,
    val_ds: TemporalQueryDataset,
    tokenizer: StoryTokenizer,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device(args.device)
    model, config = build_model(args, condition, tokenizer)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_examples)
    before = evaluate(
        model,
        val_ds,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch in loader:
            loss, _, _ = answer_loss_and_logits(
                model, tokenizer, batch, max_length=args.max_length, device=device
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        avg_loss = statistics.fmean(epoch_losses)
        history.append(avg_loss)
        print(f"[{condition} seed={seed}] epoch={epoch + 1} answer_loss={avg_loss:.4f}")
    after = evaluate(
        model,
        val_ds,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    result: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "config": {
            "use_phase_stream": getattr(config, "use_phase_stream", None),
            "use_resonance_bias": getattr(config, "use_resonance_bias", None),
            "center_resonance": getattr(config, "center_resonance", None),
            "normalize_resonance": getattr(config, "normalize_resonance", None),
            "story_phase_prior": getattr(model, "story_phase_prior", None),
        },
        "before": before,
        "after": after,
        "history": history,
        "elapsed_sec": time.perf_counter() - start,
    }
    if args.interventions:
        result["interventions"] = evaluate_interventions(
            model,
            val_ds,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            args=args,
            base=after,
        )
    return result


def summarise(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["condition"], []).append(run)
    summary = {}
    for condition, rows in grouped.items():
        acc = [float(row["after"]["answer_acc"]) for row in rows]
        margin = [float(row["after"]["answer_margin"]) for row in rows]
        loss = [float(row["after"]["answer_loss"]) for row in rows]
        summary[condition] = {
            "n": float(len(rows)),
            "params": float(rows[0]["params"]),
            "answer_acc_mean": statistics.fmean(acc),
            "answer_acc_std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
            "answer_margin_mean": statistics.fmean(margin),
            "answer_loss_mean": statistics.fmean(loss),
        }
    return summary


def write_report(path: Path, args: argparse.Namespace, runs: list[dict[str, Any]]) -> None:
    summary = summarise(runs)
    lines = [
        "# Temporal Story Query Evaluation",
        "",
        f"Train examples: `{args.train_examples}`, validation examples: `{args.val_examples}`, epochs: `{args.epochs}`.",
        "",
        "| Condition | Params | Answer Acc | Acc Std | Answer Margin | Answer Loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, row in summary.items():
        lines.append(
            f"| {condition} | {int(row['params']):,} | {row['answer_acc_mean']:.4f} | "
            f"{row['answer_acc_std']:.4f} | {row['answer_margin_mean']:.4f} | {row['answer_loss_mean']:.4f} |"
        )
    intervention_rows = []
    for condition in summary:
        condition_runs = [run for run in runs if run["condition"] == condition]
        keys = sorted(
            {
                key
                for run in condition_runs
                for key in (run.get("interventions") or {}).keys()
            }
        )
        for key in keys:
            values = [run["interventions"][key] for run in condition_runs if key in (run.get("interventions") or {})]
            intervention_rows.append(
                (
                    condition,
                    key,
                    statistics.fmean(float(v["answer_acc_delta"]) for v in values),
                    statistics.fmean(float(v["answer_margin_delta"]) for v in values),
                )
            )
    if intervention_rows:
        lines.extend(
            [
                "",
                "## Intervention Deltas",
                "",
                "| Condition | Intervention | Answer Acc Delta | Margin Delta |",
                "|---|---|---:|---:|",
            ]
        )
        for condition, key, acc_delta, margin_delta in intervention_rows:
            lines.append(f"| {condition} | {key} | {acc_delta:+.4f} | {margin_delta:+.4f} |")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/story_query_eval"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", default="standard,resonance_full,resonance_full_story_prior,bias_only")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_examples", type=int, default=512)
    parser.add_argument("--val_examples", type=int, default=256)
    parser.add_argument("--n_events", type=int, default=5)
    parser.add_argument("--edge_prob", type=float, default=0.25)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--vocab_size", type=int, default=768)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--phase_init_std", type=float, default=0.3)
    parser.add_argument("--story_phase_prior_std", type=float, default=1.2)
    parser.add_argument("--story_phase_prior_noise", type=float, default=0.05)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--dataset_seed", type=int, default=4000)
    parser.add_argument("--interventions", action="store_true")
    parser.add_argument("--phase_intervention_modes", nargs="+", default=["zero", "permute", "noise"])
    parser.add_argument("--phase_noise_sigma", type=float, default=0.5)
    parser.add_argument("--resonance_scales", type=float, nargs="+", default=[0.0, 2.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_deterministic(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = TemporalQueryDataset(
        n_examples=args.train_examples,
        seed=args.dataset_seed,
        n_events=args.n_events,
        edge_prob=args.edge_prob,
    )
    val_ds = TemporalQueryDataset(
        n_examples=args.val_examples,
        seed=args.dataset_seed + 100_000,
        n_events=args.n_events,
        edge_prob=args.edge_prob,
    )
    tokenizer = StoryTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(train_ds.all_texts() + val_ds.all_texts() + ["yes no"])
    runs = []
    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    for seed in args.seeds:
        for condition in conditions:
            runs.append(train_condition(args, condition, seed, train_ds, val_ds, tokenizer))
    results = {
        "summary": summarise(runs),
        "runs": runs,
        "config": vars(args),
        "vocab": tokenizer.word2id,
    }
    (args.output_dir / "results.json").write_text(json.dumps(jsonify(results), indent=2))
    write_report(args.output_dir / "report.md", args, runs)
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
