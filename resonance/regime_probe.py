#!/usr/bin/env python3
"""Probe whether a training regime is meaningful before ablation.

This is a small diagnostic harness inspired by Li et al. (ICLR 2026),
"On the Predictive Power of Representation Dispersion in Language Models".
It does not try to prove a resonance advantage.  It answers a prior question:

    Is this task/training setup learnable, non-saturated, and geometrically
    active enough that an architecture comparison could produce signal?

The probe trains on the temporal story-query task and tracks:

    - majority baseline vs. model answer accuracy,
    - answer loss/margin,
    - final-hidden dispersion at the answer position,
    - label-conditioned within/between dispersion,
    - dispersion of low-loss vs. high-loss slices,
    - optional push-away or squeeze auxiliary loss.

Positive ``--dispersion_lambda`` adds a push-away objective:

    total_loss = answer_loss - lambda * dispersion(hidden)

Negative values squeeze representations.  If push/squeeze has no measurable
effect, the regime is probably too weak for geometry-sensitive ablations.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from resonance.device import enable_deterministic, get_device, set_seed
from story_query_eval import (  # noqa: E402
    TemporalQueryDataset,
    StoryTokenizer,
    collate_examples,
    encode_supervised_batch,
    evaluate,
)
from story_topology_eval import build_model  # noqa: E402


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def representation_dispersion(x: torch.Tensor) -> torch.Tensor:
    """Average pairwise cosine distance for rows of ``x``."""
    if x.size(0) < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    x = F.normalize(x.float(), dim=-1)
    sim = x @ x.T
    n = x.size(0)
    off_diag = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return (1.0 - sim[off_diag]).mean()


def label_dispersion(x: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Within-label and between-centroid dispersion diagnostics."""
    x = F.normalize(x.float(), dim=-1)
    labels = labels.detach().cpu()
    unique = sorted(int(v) for v in torch.unique(labels))
    within_values: list[float] = []
    centroids: list[torch.Tensor] = []
    for label in unique:
        mask = labels == label
        group = x[mask.to(x.device)]
        if group.numel() == 0:
            continue
        centroid = F.normalize(group.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        centroids.append(centroid)
        if group.size(0) > 1:
            distances = 1.0 - (group @ centroid)
            within_values.extend(distances.detach().cpu().tolist())
    if len(centroids) >= 2:
        centroid_tensor = torch.stack(centroids, dim=0)
        between = float(representation_dispersion(centroid_tensor).item())
    else:
        between = 0.0
    within = statistics.fmean(within_values) if within_values else 0.0
    return {
        "within_label_distance": within,
        "between_label_centroid_distance": between,
        "between_minus_within": between - within,
    }


def forward_answer_batch(
    model: torch.nn.Module,
    tokenizer: StoryTokenizer,
    batch,
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return answer loss, logits, targets, hidden states, and optional phase states."""
    input_ids, _labels, answer_positions = encode_supervised_batch(
        tokenizer,
        batch,
        max_length=max_length,
        device=device,
    )
    out = model(input_ids, labels=None, return_hidden=True)
    logits = out["logits"]
    hidden = out["hidden_states"]
    yes_id = tokenizer.word2id["yes"]
    no_id = tokenizer.word2id["no"]
    row_ids = torch.arange(input_ids.size(0), device=device)
    answer_logits = logits[row_ids, answer_positions][:, [yes_id, no_id]]
    targets = torch.tensor(
        [0 if example.answer == "yes" else 1 for example in batch],
        dtype=torch.long,
        device=device,
    )
    answer_loss = F.cross_entropy(answer_logits, targets)
    answer_hidden = hidden[row_ids, answer_positions]
    phase_states = out.get("phase_states")
    answer_phase = None
    if isinstance(phase_states, torch.Tensor):
        answer_phase = phase_states[row_ids, answer_positions]
    return answer_loss, answer_logits, targets, answer_hidden, answer_phase


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Small supervised contrastive loss for phase/structural states."""
    if embeddings.size(0) < 3:
        return embeddings.new_tensor(0.0)
    embeddings = F.normalize(embeddings.float(), dim=-1)
    logits = embeddings @ embeddings.T / temperature
    eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(eye, -float("inf"))
    positive = (labels[:, None] == labels[None, :]) & ~eye
    valid = positive.sum(dim=1) > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probs = torch.where(positive, log_probs, torch.zeros_like(log_probs))
    per_row = -positive_log_probs.sum(dim=1) / positive.sum(dim=1).clamp(min=1)
    return per_row[valid].mean()


@torch.no_grad()
def evaluate_geometry(
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
    hidden_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    loss_rows: list[torch.Tensor] = []
    for batch in loader:
        _loss, answer_logits, targets, answer_hidden, _answer_phase = forward_answer_batch(
            model,
            tokenizer,
            batch,
            max_length=max_length,
            device=device,
        )
        per_example = F.cross_entropy(answer_logits, targets, reduction="none")
        hidden_rows.append(answer_hidden.detach().cpu())
        target_rows.append(targets.detach().cpu())
        loss_rows.append(per_example.detach().cpu())
    hidden = torch.cat(hidden_rows, dim=0)
    targets = torch.cat(target_rows, dim=0)
    losses = torch.cat(loss_rows, dim=0)

    order = torch.argsort(losses)
    n = max(1, hidden.size(0) // 4)
    easy = hidden[order[:n]]
    hard = hidden[order[-n:]]
    diagnostics = {
        "dispersion_all": float(representation_dispersion(hidden).item()),
        "dispersion_easy_quartile": float(representation_dispersion(easy).item()),
        "dispersion_hard_quartile": float(representation_dispersion(hard).item()),
        "hard_minus_easy_dispersion": float(
            representation_dispersion(hard).item() - representation_dispersion(easy).item()
        ),
    }
    diagnostics.update(label_dispersion(hidden, targets))
    return diagnostics


def label_balance(dataset: TemporalQueryDataset) -> dict[str, float]:
    yes = sum(1 for example in dataset.examples if example.answer == "yes")
    no = len(dataset) - yes
    majority = max(yes, no) / max(len(dataset), 1)
    return {
        "yes_fraction": yes / max(len(dataset), 1),
        "no_fraction": no / max(len(dataset), 1),
        "majority_acc": majority,
    }


def train_condition(
    args: argparse.Namespace,
    condition: str,
    seed: int,
    train_ds: TemporalQueryDataset,
    val_ds: TemporalQueryDataset,
    tokenizer: StoryTokenizer,
) -> dict[str, Any]:
    set_seed(seed)
    model_condition = condition
    phase_contrastive = False
    if model_condition.endswith("_phase_contrastive"):
        phase_contrastive = True
        model_condition = model_condition.removesuffix("_phase_contrastive")
    phase_contrastive_weight = args.phase_contrastive_weight
    if phase_contrastive and phase_contrastive_weight == 0.0:
        phase_contrastive_weight = 0.1
    device = get_device(args.device)
    model, config = build_model(args, model_condition, tokenizer)
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
    before_geometry = evaluate_geometry(
        model,
        val_ds,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        answer_losses = []
        dispersions = []
        phase_aux_losses = []
        for batch in loader:
            answer_loss, _answer_logits, targets, hidden, answer_phase = forward_answer_batch(
                model,
                tokenizer,
                batch,
                max_length=args.max_length,
                device=device,
            )
            dispersion = representation_dispersion(hidden)
            phase_aux = hidden.new_tensor(0.0)
            if (phase_contrastive or phase_contrastive_weight > 0.0) and answer_phase is not None:
                phase_aux = supervised_contrastive_loss(
                    answer_phase,
                    targets,
                    temperature=args.phase_contrastive_temperature,
                )
            total_loss = (
                answer_loss
                + phase_contrastive_weight * phase_aux
                - args.dispersion_lambda * dispersion
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(total_loss.item()))
            answer_losses.append(float(answer_loss.item()))
            dispersions.append(float(dispersion.item()))
            phase_aux_losses.append(float(phase_aux.item()))
        row = {
            "epoch": float(epoch + 1),
            "total_loss": statistics.fmean(losses),
            "answer_loss": statistics.fmean(answer_losses),
            "train_dispersion": statistics.fmean(dispersions),
            "phase_aux_loss": statistics.fmean(phase_aux_losses),
        }
        if getattr(args, "eval_each_epoch", False):
            val_metrics = evaluate(
                model,
                val_ds,
                tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        message = (
            f"[{condition} seed={seed}] epoch={epoch + 1} "
            f"answer_loss={row['answer_loss']:.4f} disp={row['train_dispersion']:.4f}"
        )
        if getattr(args, "eval_each_epoch", False):
            message += (
                f" val_acc={row['val_answer_acc']:.4f} "
                f"val_loss={row['val_answer_loss']:.4f}"
            )
        print(message)

    after = evaluate(
        model,
        val_ds,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    after_geometry = evaluate_geometry(
        model,
        val_ds,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    checkpoint_path = None
    if getattr(args, "save_models", False):
        checkpoint_dir = args.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        safe_condition = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in condition)
        checkpoint_path = checkpoint_dir / f"{safe_condition}_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "condition": condition,
                "model_condition": model_condition,
                "seed": seed,
                "args": jsonify(vars(args)),
                "model_config": jsonify(getattr(config, "__dict__", {})),
                "tokenizer": {
                    "vocab_size": tokenizer.vocab_size,
                    "word2id": tokenizer.word2id,
                },
            },
            checkpoint_path,
        )
    return {
        "condition": condition,
        "model_condition": model_condition,
        "phase_contrastive": phase_contrastive or phase_contrastive_weight > 0.0,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "config": {
            "use_phase_stream": getattr(config, "use_phase_stream", None),
            "use_resonance_bias": getattr(config, "use_resonance_bias", None),
            "init_preset": getattr(config, "init_preset", None),
            "resonance_kernel": getattr(config, "resonance_kernel", None),
            "bias_mode": getattr(config, "bias_mode", None),
            "phase_update_mode": getattr(config, "phase_update_mode", None),
            "phase_condition_qk": getattr(config, "phase_condition_qk", None),
            "n_structural_heads": getattr(config, "n_structural_heads", None),
            "relation_value_mode": getattr(config, "relation_value_mode", None),
            "attention_variant": getattr(config, "attention_variant", None),
            "dispersion_lambda": args.dispersion_lambda,
            "phase_contrastive_weight": phase_contrastive_weight,
        },
        "before": before,
        "after": after,
        "before_geometry": before_geometry,
        "after_geometry": after_geometry,
        "history": history,
    }


def summarise(runs: list[dict[str, Any]], majority_acc: float) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["condition"], []).append(run)
    summary = {}
    for condition, rows in grouped.items():
        acc = [float(row["after"]["answer_acc"]) for row in rows]
        loss_gain = [
            float(row["before"]["answer_loss"]) - float(row["after"]["answer_loss"])
            for row in rows
        ]
        disp_delta = [
            float(row["after_geometry"]["dispersion_all"]) - float(row["before_geometry"]["dispersion_all"])
            for row in rows
        ]
        label_gap = [float(row["after_geometry"]["between_minus_within"]) for row in rows]
        summary[condition] = {
            "n": float(len(rows)),
            "params": float(rows[0]["params"]),
            "answer_acc_mean": statistics.fmean(acc),
            "answer_acc_std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
            "acc_minus_majority": statistics.fmean(acc) - majority_acc,
            "answer_loss_gain": statistics.fmean(loss_gain),
            "dispersion_delta": statistics.fmean(disp_delta),
            "label_gap": statistics.fmean(label_gap),
        }
    return summary


def regime_verdict(summary: dict[str, dict[str, float]]) -> str:
    rows = list(summary.values())
    if not rows:
        return "no_runs"
    best_acc = max(row["answer_acc_mean"] for row in rows)
    best_acc_gain = max(row["acc_minus_majority"] for row in rows)
    if best_acc_gain < 0.05:
        return "weak: no condition beats majority baseline by 5 points"
    if best_acc > 0.90:
        return "weak: task is close to saturated"
    viable = [
        row
        for row in rows
        if row["acc_minus_majority"] >= 0.05 and row["answer_loss_gain"] >= 0.02
    ]
    if not viable:
        return "weak: training barely reduces answer loss"
    if max(row["label_gap"] for row in viable) <= 0:
        return "weak: behavior improves but hidden geometry is not label-separated"
    return "candidate: learnable and non-saturated"


def write_report(
    path: Path,
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
    balance: dict[str, float],
) -> None:
    summary = summarise(runs, balance["majority_acc"])
    lines = [
        "# Regime Probe",
        "",
        f"Train examples: `{args.train_examples}`, validation examples: `{args.val_examples}`, epochs: `{args.epochs}`.",
        f"Dispersion lambda: `{args.dispersion_lambda}`.",
        "",
        f"Validation label balance: yes `{balance['yes_fraction']:.3f}`, no `{balance['no_fraction']:.3f}`, majority acc `{balance['majority_acc']:.3f}`.",
        "",
        f"Verdict: **{regime_verdict(summary)}**.",
        "",
        "| Condition | Params | Acc | Acc-Maj | Loss Gain | Disp Delta | Label Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in summary.items():
        lines.append(
            f"| {condition} | {int(row['params']):,} | {row['answer_acc_mean']:.4f} | "
            f"{row['acc_minus_majority']:+.4f} | {row['answer_loss_gain']:+.4f} | "
            f"{row['dispersion_delta']:+.4f} | {row['label_gap']:+.4f} |"
        )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/regime_probe"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", default="standard,resonance_full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_examples", type=int, default=256)
    parser.add_argument("--val_examples", type=int, default=128)
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
    parser.add_argument("--dispersion_lambda", type=float, default=0.0)
    parser.add_argument("--phase_contrastive_weight", type=float, default=0.0)
    parser.add_argument("--phase_contrastive_temperature", type=float, default=0.2)
    parser.add_argument("--eval_each_epoch", action="store_true")
    parser.add_argument("--save_models", action="store_true", help="Save trained model checkpoints for interpretability audits.")
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
    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    runs = []
    for seed in args.seeds:
        for condition in conditions:
            runs.append(train_condition(args, condition, seed, train_ds, val_ds, tokenizer))
    balance = label_balance(val_ds)
    results = {
        "summary": summarise(runs, balance["majority_acc"]),
        "verdict": regime_verdict(summarise(runs, balance["majority_acc"])),
        "label_balance": balance,
        "runs": runs,
        "config": vars(args),
    }
    (args.output_dir / "results.json").write_text(json.dumps(jsonify(results), indent=2))
    write_report(args.output_dir / "report.md", args, runs, balance)
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
