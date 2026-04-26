#!/usr/bin/env python3
"""Topology-bearing natural-language story evaluation.

This harness trains standard/resonance transformers on generated stories where
several surface forms share one semantic graph.  It measures whether model
representations retrieve same-graph paraphrases and separate hard negatives
whose wording is similar but topology differs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from resonance.config import ResonanceConfig, StandardConfig
from resonance.device import enable_deterministic, get_device, set_seed
from resonance.interpretability import (
    phase_intervention,
    resonance_diagnostics,
    resonance_weight_scale,
    trace_resonance_forward,
)
from resonance.models import ResonanceTransformer, StandardTransformer
from synthetic.semantic_story import (
    NAMES,
    OBJECTS,
    PLACES,
    PREDICATES,
    VERB_FORMS,
    StoryGraphDataset,
    StoryTokenizer,
)


def _base_condition(condition: str) -> tuple[str, bool, bool, bool]:
    story_prior = False
    centered = False
    normalized = False
    base = condition
    if base.endswith("_story_prior"):
        story_prior = True
        base = base.removesuffix("_story_prior")
    if base.endswith("_normalized"):
        normalized = True
        centered = True
        base = base.removesuffix("_normalized")
    if base.endswith("_centered"):
        centered = True
        base = base.removesuffix("_centered")
    return base, story_prior, centered, normalized


def story_role_word_groups() -> dict[str, list[str]]:
    """Words with known structural roles in the generated-story language."""
    verb_words = sorted(
        set(PREDICATES)
        | {form for forms in VERB_FORMS.values() for form in forms}
        | {"did", "not"}
    )
    return {
        "person": [name.lower() for name in NAMES],
        "object": OBJECTS,
        "place": PLACES,
        "event_verb": verb_words,
        "temporal": ["after", "order"],
        "causal": ["because"],
        "enabling": ["possible"],
        "preventing": ["even"],
        "belief": ["thought"],
        "argument_marker": ["to", "in", "the"],
        "frame": ["together", "something", "happened", "mattered"],
    }


def apply_story_phase_prior(
    model: ResonanceTransformer,
    tokenizer: StoryTokenizer,
    *,
    std: float,
    noise: float,
    seed: int,
) -> dict[str, int]:
    """Initialize phase rows from known story-language structural roles."""
    groups = story_role_word_groups()
    generator = torch.Generator(device=model.embedding.phase.weight.device)
    generator.manual_seed(seed)
    assigned: dict[str, int] = {}
    with torch.no_grad():
        model.embedding.phase.weight.normal_(mean=0.0, std=std, generator=generator)
        for role_index, (role, words) in enumerate(groups.items()):
            base = torch.randn(
                model.config.n_frequencies,
                generator=generator,
                device=model.embedding.phase.weight.device,
            ) * std
            count = 0
            for word in words:
                token_id = tokenizer.word2id.get(word)
                if token_id is None:
                    continue
                row_noise = torch.randn(
                    model.config.n_frequencies,
                    generator=generator,
                    device=model.embedding.phase.weight.device,
                ) * noise
                model.embedding.phase.weight[token_id].copy_(base + row_noise)
                count += 1
            assigned[role] = count
    return assigned


def phase_role_separation(
    model: torch.nn.Module,
    tokenizer: StoryTokenizer,
) -> dict[str, float]:
    """Compare mean resonance for same-role vs different-role vocabulary rows."""
    if not isinstance(model, ResonanceTransformer):
        return {}
    groups = story_role_word_groups()
    role_ids: dict[str, list[int]] = {
        role: [tokenizer.word2id[word] for word in words if word in tokenizer.word2id]
        for role, words in groups.items()
    }
    role_ids = {role: ids for role, ids in role_ids.items() if len(ids) >= 2}
    if len(role_ids) < 2:
        return {}
    phase = model.embedding.phase.weight.detach()
    same_values: list[float] = []
    diff_values: list[float] = []
    roles = list(role_ids)
    for role, ids in role_ids.items():
        rows = phase[ids]
        diff = rows.unsqueeze(1) - rows.unsqueeze(0)
        resonance = torch.cos(diff).mean(dim=-1)
        mask = ~torch.eye(len(ids), dtype=torch.bool, device=resonance.device)
        same_values.extend(resonance[mask].detach().cpu().tolist())
    for i, role_a in enumerate(roles):
        for role_b in roles[i + 1 :]:
            rows_a = phase[role_ids[role_a]]
            rows_b = phase[role_ids[role_b]]
            diff = rows_a.unsqueeze(1) - rows_b.unsqueeze(0)
            resonance = torch.cos(diff).mean(dim=-1)
            diff_values.extend(resonance.flatten().detach().cpu().tolist())
    same_mean = statistics.fmean(same_values)
    diff_mean = statistics.fmean(diff_values)
    return {
        "phase_same_role_resonance": same_mean,
        "phase_diff_role_resonance": diff_mean,
        "phase_role_gap": same_mean - diff_mean,
    }


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


def build_model(args: argparse.Namespace, condition: str, tokenizer: StoryTokenizer):
    base_condition, story_prior, centered, normalized = _base_condition(condition)
    common = {
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": args.max_length,
        "embed_dim": args.embed_dim,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "gradient_accumulation": 1,
    }
    if base_condition == "standard":
        config = StandardConfig(name=condition, **common)
        return StandardTransformer(config), config
    if base_condition not in {"resonance_full", "phase_stream_only", "bias_only", "resonance_inert"}:
        raise ValueError(f"unknown condition: {condition}")
    config = ResonanceConfig(
        name=condition,
        n_frequencies=args.n_frequencies,
        phase_init_std=args.phase_init_std,
        resonance_attn_weight=args.resonance_attn_weight,
        resonance_blend=args.resonance_blend,
        use_phase_stream=base_condition in {"resonance_full", "phase_stream_only"},
        use_resonance_bias=base_condition in {"resonance_full", "bias_only"},
        center_resonance=centered,
        normalize_resonance=normalized,
        **common,
    )
    model = ResonanceTransformer(config)
    if story_prior:
        assigned = apply_story_phase_prior(
            model,
            tokenizer,
            std=args.story_phase_prior_std,
            noise=args.story_phase_prior_noise,
            seed=args.dataset_seed,
        )
        model.story_phase_prior = assigned  # type: ignore[attr-defined]
    return model, config


def pooled_hidden(model: torch.nn.Module, tokens: torch.Tensor, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(tokens, labels=tokens, return_hidden=True)
    hidden = out["hidden_states"]
    mask = (tokens != pad_id).float().unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled, out["loss"]


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=-1)
    sim = embeddings @ embeddings.T / temperature
    n = sim.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, -float("inf"))
    positive = labels[:, None] == labels[None, :]
    positive = positive & ~eye
    valid = positive.sum(dim=1) > 0
    if not valid.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    log_probs = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positive_log_probs = torch.where(positive, log_probs, torch.zeros_like(log_probs))
    loss = -positive_log_probs.sum(dim=1) / positive.sum(dim=1).clamp(min=1)
    return loss[valid].mean()


def flatten_batch(batch: dict[str, object]) -> tuple[list[str], torch.Tensor]:
    texts: list[str] = []
    labels: list[int] = []
    unique = 1_000_000
    for group_id, positives in enumerate(batch["positives"]):  # type: ignore[union-attr]
        for text in positives:
            texts.append(text)
            labels.append(group_id)
    for negatives in batch["hard_negatives"]:  # type: ignore[union-attr]
        for text in negatives:
            texts.append(text)
            labels.append(unique)
            unique += 1
    return texts, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: StoryGraphDataset,
    tokenizer: StoryTokenizer,
    device: torch.device,
    max_groups: int,
    max_length: int,
) -> dict[str, float]:
    model.eval()
    texts: list[str] = []
    labels: list[int] = []
    negative_by_group: dict[int, list[int]] = {}
    for group_id in range(min(max_groups, len(dataset))):
        row = dataset[group_id]
        for text in row["positives"]:  # type: ignore[union-attr]
            labels.append(group_id)
            texts.append(text)
        negative_by_group[group_id] = []
        for text in row["hard_negatives"]:  # type: ignore[union-attr]
            labels.append(-(group_id + 1))
            negative_by_group[group_id].append(len(texts))
            texts.append(text)

    tokens = tokenizer.batch_encode(texts, max_length=max_length, device=device)
    embeddings, lm_loss = pooled_hidden(model, tokens, tokenizer.pad_id)
    embeddings = F.normalize(embeddings, dim=-1)
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(-float("inf"))
    nearest = sim.argmax(dim=1).detach().cpu().tolist()

    positive_indices = [i for i, label in enumerate(labels) if label >= 0]
    same_graph_hits = sum(
        1 for i in positive_indices if labels[nearest[i]] == labels[i]
    )
    positive_group_acc = same_graph_hits / max(len(positive_indices), 1)

    hard_margin_values = []
    for i in positive_indices:
        group_id = labels[i]
        same = [j for j, label in enumerate(labels) if label == group_id and j != i]
        hard = negative_by_group.get(group_id, [])
        if same and hard:
            hard_margin_values.append(
                float(sim[i, same].mean().item() - sim[i, hard].mean().item())
            )

    return {
        "same_graph_retrieval_acc": positive_group_acc,
        "hard_negative_margin": statistics.fmean(hard_margin_values) if hard_margin_values else 0.0,
        "lm_loss": float(lm_loss.item()),
    }


@torch.no_grad()
def collect_resonance_diagnostics(
    model: torch.nn.Module,
    dataset: StoryGraphDataset,
    tokenizer: StoryTokenizer,
    device: torch.device,
    max_length: int,
) -> dict[str, float]:
    """Collect one small resonance trace for interpretability sanity checks."""
    if not isinstance(model, ResonanceTransformer):
        return {}
    texts: list[str] = []
    for group_id in range(min(8, len(dataset))):
        row = dataset[group_id]
        positives = row["positives"]  # type: ignore[assignment]
        texts.append(positives[0])
    tokens = tokenizer.batch_encode(texts, max_length=max_length, device=device)
    trace = trace_resonance_forward(model, tokens)
    return resonance_diagnostics(trace)


@torch.no_grad()
def evaluate_interventions(
    model: torch.nn.Module,
    dataset: StoryGraphDataset,
    tokenizer: StoryTokenizer,
    device: torch.device,
    args: argparse.Namespace,
    base: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Measure causal sensitivity to phase and resonance perturbations."""
    if not isinstance(model, ResonanceTransformer):
        return {}
    metrics: dict[str, dict[str, float]] = {}
    for mode in args.phase_intervention_modes:
        with phase_intervention(model, mode=mode, sigma=args.phase_noise_sigma, seed=args.dataset_seed):
            value = evaluate(model, dataset, tokenizer, device, args.eval_graphs, args.max_length)
        metrics[f"phase_{mode}"] = {
            **value,
            "same_graph_retrieval_delta": value["same_graph_retrieval_acc"] - base["same_graph_retrieval_acc"],
            "hard_negative_margin_delta": value["hard_negative_margin"] - base["hard_negative_margin"],
            "lm_loss_delta": value["lm_loss"] - base["lm_loss"],
        }
    for scale in args.resonance_scales:
        with resonance_weight_scale(model, scale):
            value = evaluate(model, dataset, tokenizer, device, args.eval_graphs, args.max_length)
        key = f"resonance_scale_{scale:g}"
        metrics[key] = {
            **value,
            "same_graph_retrieval_delta": value["same_graph_retrieval_acc"] - base["same_graph_retrieval_acc"],
            "hard_negative_margin_delta": value["hard_negative_margin"] - base["hard_negative_margin"],
            "lm_loss_delta": value["lm_loss"] - base["lm_loss"],
        }
    return metrics


def train_condition(
    args: argparse.Namespace,
    condition: str,
    seed: int,
    train_ds: StoryGraphDataset,
    val_ds: StoryGraphDataset,
    tokenizer: StoryTokenizer,
) -> dict[str, object]:
    set_seed(seed)
    device = get_device(args.device)
    model, config = build_model(args, condition, tokenizer)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=StoryGraphDataset.collate_fn,
    )
    before = evaluate(model, val_ds, tokenizer, device, args.eval_graphs, args.max_length)
    history = {"lm_loss": [], "contrastive_loss": [], "total_loss": []}
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_lm = 0.0
        epoch_ctr = 0.0
        epoch_total = 0.0
        n_batches = 0
        for batch in loader:
            texts, labels = flatten_batch(batch)
            labels = labels.to(device)
            tokens = tokenizer.batch_encode(texts, max_length=args.max_length, device=device)
            embeddings, lm_loss = pooled_hidden(model, tokens, tokenizer.pad_id)
            ctr_loss = supervised_contrastive_loss(embeddings, labels, args.temperature)
            total_loss = args.lm_weight * lm_loss + args.contrastive_weight * ctr_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_lm += float(lm_loss.item())
            epoch_ctr += float(ctr_loss.item())
            epoch_total += float(total_loss.item())
            n_batches += 1
        history["lm_loss"].append(epoch_lm / max(n_batches, 1))
        history["contrastive_loss"].append(epoch_ctr / max(n_batches, 1))
        history["total_loss"].append(epoch_total / max(n_batches, 1))
        print(
            f"[{condition} seed={seed}] epoch={epoch + 1} "
            f"lm={history['lm_loss'][-1]:.4f} ctr={history['contrastive_loss'][-1]:.4f}"
        )
    after = evaluate(model, val_ds, tokenizer, device, args.eval_graphs, args.max_length)
    result: dict[str, object] = {
        "condition": condition,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "config": {
            "use_phase_stream": getattr(config, "use_phase_stream", None),
            "use_resonance_bias": getattr(config, "use_resonance_bias", None),
            "resonance_attn_weight": getattr(config, "resonance_attn_weight", None),
            "center_resonance": getattr(config, "center_resonance", None),
            "normalize_resonance": getattr(config, "normalize_resonance", None),
            "story_phase_prior": getattr(model, "story_phase_prior", None),
        },
        "before": before,
        "after": after,
        "history": history,
        "elapsed_sec": time.perf_counter() - start,
    }
    if isinstance(model, ResonanceTransformer):
        result["diagnostics"] = collect_resonance_diagnostics(
            model, val_ds, tokenizer, device, args.max_length
        )
        result["phase_role_separation"] = phase_role_separation(model, tokenizer)
        if args.interventions:
            result["interventions"] = evaluate_interventions(
                model, val_ds, tokenizer, device, args, after
            )
    if args.save_models:
        checkpoint_path = args.output_dir / f"{condition}_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "condition": condition,
                "seed": seed,
                "config": config,
                "tokenizer_word2id": tokenizer.word2id,
            },
            checkpoint_path,
        )
        result["checkpoint"] = str(checkpoint_path)
    return result


def summarise(runs: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        grouped.setdefault(str(run["condition"]), []).append(run)
    summary = {}
    for condition, rows in grouped.items():
        retrieval = [float(row["after"]["same_graph_retrieval_acc"]) for row in rows]  # type: ignore[index]
        margin = [float(row["after"]["hard_negative_margin"]) for row in rows]  # type: ignore[index]
        summary[condition] = {
            "n": len(rows),
            "params": rows[0]["params"],
            "same_graph_retrieval_mean": statistics.fmean(retrieval),
            "same_graph_retrieval_std": statistics.stdev(retrieval) if len(retrieval) > 1 else 0.0,
            "hard_negative_margin_mean": statistics.fmean(margin),
        }
    return summary


def write_report(path: Path, args: argparse.Namespace, runs: list[dict[str, object]]) -> None:
    summary = summarise(runs)
    lines = [
        "# Story Graph Topology Evaluation",
        "",
        f"Train graphs: `{args.train_graphs}`, validation graphs: `{args.val_graphs}`, epochs: `{args.epochs}`.",
        "",
        "| Condition | Params | Same-Graph Retrieval | Retrieval Std | Hard-Negative Margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, row in summary.items():  # type: ignore[assignment]
        lines.append(
            f"| {condition} | {row['params']:,} | {row['same_graph_retrieval_mean']:.4f} | "
            f"{row['same_graph_retrieval_std']:.4f} | {row['hard_negative_margin_mean']:.4f} |"
        )
    lines.append("")
    lines.append("Positive examples share a semantic graph; hard negatives mutate one structural fact.")
    intervention_rows: list[tuple[str, str, float, float, float]] = []
    diagnostic_rows: list[tuple[str, float, float, float]] = []
    for condition in summary:
        condition_runs = [run for run in runs if run["condition"] == condition]
        intervention_keys = sorted(
            {
                key
                for run in condition_runs
                for key in (run.get("interventions") or {}).keys()  # type: ignore[union-attr]
            }
        )
        for key in intervention_keys:
            values = [
                run["interventions"][key]  # type: ignore[index]
                for run in condition_runs
                if key in (run.get("interventions") or {})
            ]
            intervention_rows.append(
                (
                    condition,
                    key,
                    statistics.fmean(float(v["same_graph_retrieval_delta"]) for v in values),
                    statistics.fmean(float(v["hard_negative_margin_delta"]) for v in values),
                    statistics.fmean(float(v["lm_loss_delta"]) for v in values),
                )
            )
        diagnostics = [run.get("diagnostics") for run in condition_runs if run.get("diagnostics")]
        if diagnostics:
            diagnostic_rows.append(
                (
                    condition,
                    statistics.fmean(float(d["resonance_effective_rank_first"]) for d in diagnostics),  # type: ignore[index]
                    statistics.fmean(float(d["attention_delta_abs_mean"]) for d in diagnostics),  # type: ignore[index]
                    statistics.fmean(float(d["resonance_logit_abs_mean"]) for d in diagnostics),  # type: ignore[index]
                )
            )
    if intervention_rows:
        lines.extend(
            [
                "",
                "## Intervention Deltas",
                "",
                "| Condition | Intervention | Retrieval Delta | Margin Delta | LM Loss Delta |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for condition, key, retrieval_delta, margin_delta, lm_delta in intervention_rows:
            lines.append(
                f"| {condition} | {key} | {retrieval_delta:+.4f} | {margin_delta:+.4f} | {lm_delta:+.4f} |"
            )
    if diagnostic_rows:
        lines.extend(
            [
                "",
                "## Resonance Diagnostics",
                "",
                "| Condition | Effective Rank | Attention Delta Abs Mean | Resonance Logit Abs Mean |",
                "|---|---:|---:|---:|",
            ]
        )
        for condition, rank, delta_abs, logit_abs in diagnostic_rows:
            lines.append(
                f"| {condition} | {rank:.4f} | {delta_abs:.6f} | {logit_abs:.6f} |"
            )
    role_rows = [
        (
            str(run["condition"]),
            run.get("phase_role_separation"),
        )
        for run in runs
        if run.get("phase_role_separation")
    ]
    if role_rows:
        grouped_roles: dict[str, list[dict[str, float]]] = {}
        for condition, value in role_rows:
            grouped_roles.setdefault(condition, []).append(value)  # type: ignore[arg-type]
        lines.extend(
            [
                "",
                "## Phase Role Separation",
                "",
                "| Condition | Same-Role R | Diff-Role R | Gap |",
                "|---|---:|---:|---:|",
            ]
        )
        for condition, values in grouped_roles.items():
            same = statistics.fmean(float(v["phase_same_role_resonance"]) for v in values)
            diff = statistics.fmean(float(v["phase_diff_role_resonance"]) for v in values)
            gap = statistics.fmean(float(v["phase_role_gap"]) for v in values)
            lines.append(f"| {condition} | {same:.4f} | {diff:.4f} | {gap:.4f} |")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/story_topology_eval"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", default="standard,resonance_full,bias_only,resonance_inert")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_graphs", type=int, default=128)
    parser.add_argument("--val_graphs", type=int, default=64)
    parser.add_argument("--eval_graphs", type=int, default=48)
    parser.add_argument("--variants_per_graph", type=int, default=4)
    parser.add_argument("--hard_negatives_per_graph", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--phase_init_std", type=float, default=0.3)
    parser.add_argument("--story_phase_prior_std", type=float, default=1.2)
    parser.add_argument("--story_phase_prior_noise", type=float, default=0.05)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lm_weight", type=float, default=0.5)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--dataset_seed", type=int, default=2000)
    parser.add_argument("--interventions", action="store_true")
    parser.add_argument("--phase_intervention_modes", nargs="+", default=["zero", "permute", "noise"])
    parser.add_argument("--phase_noise_sigma", type=float, default=0.5)
    parser.add_argument("--resonance_scales", type=float, nargs="+", default=[0.0, 2.0])
    parser.add_argument("--save_models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_deterministic(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = StoryGraphDataset(
        n_graphs=args.train_graphs,
        variants_per_graph=args.variants_per_graph,
        hard_negatives_per_graph=args.hard_negatives_per_graph,
        seed=args.dataset_seed,
    )
    val_ds = StoryGraphDataset(
        n_graphs=args.val_graphs,
        variants_per_graph=args.variants_per_graph,
        hard_negatives_per_graph=args.hard_negatives_per_graph,
        seed=args.dataset_seed + 10_000,
    )
    tokenizer = StoryTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(train_ds.all_texts() + val_ds.all_texts())
    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    runs = []
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
