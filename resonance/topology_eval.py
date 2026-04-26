#!/usr/bin/env python3
"""Topology-bearing proof-equivalence evaluation.

This is a closer test of the resonance hypothesis than generic synthetic LM:
positive pairs are programs that reduce to the same beta-normal form.  Models
are trained with language-modeling plus supervised contrastive loss, then
evaluated by held-out same-equivalence-class retrieval.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from resonance.config import ResonanceConfig, StandardConfig
from resonance.device import enable_deterministic, get_device, set_seed
from resonance.models import ResonanceTransformer, StandardTransformer
from synthetic.contrastive_trainer import ProgramTokenizer
from synthetic.lambda_generator import normal_form
from synthetic.proof_walk_generator import CrawlMode, ProofWalkDataset


class NormalFormProofDataset(Dataset):
    """Proof groups filtered to one beta-normal form and deduplicated."""

    def __init__(
        self,
        *,
        requested_groups: int,
        programs_per_group: int,
        min_programs: int,
        max_term_depth: int,
        max_term_size: int,
        seed: int,
        crawl_mode: CrawlMode = CrawlMode.HYBRID,
        oversample: int = 8,
    ) -> None:
        self.groups: list[dict[str, Any]] = []
        raw = ProofWalkDataset(
            n_groups=requested_groups * oversample,
            programs_per_group=programs_per_group,
            max_term_depth=max_term_depth,
            max_term_size=max_term_size,
            generator_seed=seed,
            crawl_mode=crawl_mode,
            structured_ratio=0.5,
        )
        for i in range(len(raw)):
            full = raw.get_full_group(i)
            buckets: dict[str, list[str]] = {}
            for term in full["program_terms"]:
                nf = normal_form(term).to_string()
                buckets.setdefault(nf, [])
                program = term.to_string()
                if program not in buckets[nf]:
                    buckets[nf].append(program)
            if not buckets:
                continue
            nf, programs = max(buckets.items(), key=lambda kv: len(kv[1]))
            if len(programs) < min_programs:
                continue
            self.groups.append(
                {
                    "statement": full["statement"],
                    "normal_form": nf,
                    "programs": programs[:programs_per_group],
                }
            )
            if len(self.groups) >= requested_groups:
                break

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.groups[idx]

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "programs": [row["programs"] for row in batch],
            "normal_forms": [row["normal_form"] for row in batch],
            "statements": [row["statement"] for row in batch],
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


def build_model(args: argparse.Namespace, condition: str, tokenizer: ProgramTokenizer):
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
    if condition == "standard":
        config = StandardConfig(name=condition, **common)
        return StandardTransformer(config), config

    if condition not in {
        "resonance_full",
        "phase_stream_only",
        "bias_only",
        "resonance_inert",
    }:
        raise ValueError(f"unknown condition: {condition}")

    config = ResonanceConfig(
        name=condition,
        n_frequencies=args.n_frequencies,
        phase_init_std=args.phase_init_std,
        resonance_attn_weight=args.resonance_attn_weight,
        resonance_blend=args.resonance_blend,
        use_phase_stream=condition in {"resonance_full", "phase_stream_only"},
        use_resonance_bias=condition in {"resonance_full", "bias_only"},
        **common,
    )
    return ResonanceTransformer(config), config


def pooled_hidden(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(tokens, labels=tokens, return_hidden=True)
    hidden = out["hidden_states"]
    mask = (tokens != pad_id).float().unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled, out["loss"]


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    group_sizes: list[int],
    temperature: float,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=-1)
    sim = embeddings @ embeddings.T / temperature
    n = sim.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(eye, -float("inf"))
    positive = torch.zeros(n, n, dtype=torch.bool, device=sim.device)
    offset = 0
    for size in group_sizes:
        positive[offset : offset + size, offset : offset + size] = True
        offset += size
    positive = positive & ~eye
    log_probs = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positive_counts = positive.sum(dim=1).clamp(min=1)
    valid = positive.sum(dim=1) > 0
    positive_log_probs = torch.where(positive, log_probs, torch.zeros_like(log_probs))
    loss = -positive_log_probs.sum(dim=1) / positive_counts
    return loss[valid].mean()


def flatten_batch(batch: dict[str, Any]) -> tuple[list[str], list[int]]:
    programs = []
    sizes = []
    for group in batch["programs"]:
        programs.extend(group)
        sizes.append(len(group))
    return programs, sizes


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: NormalFormProofDataset,
    tokenizer: ProgramTokenizer,
    device: torch.device,
    max_groups: int,
) -> dict[str, float]:
    model.eval()
    programs = []
    group_ids = []
    for group_idx in range(min(max_groups, len(dataset))):
        group = dataset[group_idx]["programs"]
        for program in group:
            programs.append(program)
            group_ids.append(group_idx)
    tokens = tokenizer.batch_encode(programs, device=str(device))
    embeddings, lm_loss = pooled_hidden(model, tokens, tokenizer.pad_id)
    embeddings = F.normalize(embeddings, dim=-1)
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(-float("inf"))
    nearest = sim.argmax(dim=1).detach().cpu().tolist()
    correct = sum(1 for i, j in enumerate(nearest) if group_ids[i] == group_ids[j])

    group_tensor = torch.tensor(group_ids, device=sim.device)
    pos_vals = []
    neg_vals = []
    for i in range(sim.size(0)):
        same = group_tensor == group_ids[i]
        same[i] = False
        diff = ~same
        diff[i] = False
        if same.any():
            pos_vals.append(float(sim[i][same].mean().item()))
        if diff.any():
            neg_vals.append(float(sim[i][diff].mean().item()))
    return {
        "nearest_same_nf_acc": correct / max(len(group_ids), 1),
        "mean_positive_similarity": statistics.fmean(pos_vals),
        "mean_negative_similarity": statistics.fmean(neg_vals),
        "pos_neg_gap": statistics.fmean(pos_vals) - statistics.fmean(neg_vals),
        "lm_loss": float(lm_loss.item()),
    }


def train_condition(
    args: argparse.Namespace,
    condition: str,
    seed: int,
    train_ds: NormalFormProofDataset,
    val_ds: NormalFormProofDataset,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device(args.device)
    tokenizer = ProgramTokenizer(max_length=args.max_length)
    model, config = build_model(args, condition, tokenizer)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=NormalFormProofDataset.collate_fn,
    )

    before = evaluate(model, val_ds, tokenizer, device, args.eval_groups)
    history = {"lm_loss": [], "contrastive_loss": [], "total_loss": []}
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_lm = 0.0
        epoch_ctr = 0.0
        epoch_total = 0.0
        batches = 0
        for batch in loader:
            programs, sizes = flatten_batch(batch)
            tokens = tokenizer.batch_encode(programs, device=str(device))
            embeddings, lm_loss = pooled_hidden(model, tokens, tokenizer.pad_id)
            ctr_loss = supervised_contrastive_loss(embeddings, sizes, args.temperature)
            total_loss = args.lm_weight * lm_loss + args.contrastive_weight * ctr_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_lm += float(lm_loss.item())
            epoch_ctr += float(ctr_loss.item())
            epoch_total += float(total_loss.item())
            batches += 1
        history["lm_loss"].append(epoch_lm / max(batches, 1))
        history["contrastive_loss"].append(epoch_ctr / max(batches, 1))
        history["total_loss"].append(epoch_total / max(batches, 1))
        print(
            f"[{condition} seed={seed}] epoch={epoch + 1} "
            f"lm={history['lm_loss'][-1]:.4f} ctr={history['contrastive_loss'][-1]:.4f}"
        )
    elapsed = time.perf_counter() - start
    after = evaluate(model, val_ds, tokenizer, device, args.eval_groups)
    return {
        "condition": condition,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "config": {
            "embed_dim": args.embed_dim,
            "layers": args.layers,
            "heads": args.heads,
            "ff_dim": args.ff_dim,
            "max_length": args.max_length,
            "lm_weight": args.lm_weight,
            "contrastive_weight": args.contrastive_weight,
            "temperature": args.temperature,
            "resonance_attn_weight": getattr(config, "resonance_attn_weight", None),
            "phase_init_std": getattr(config, "phase_init_std", None),
            "use_phase_stream": getattr(config, "use_phase_stream", None),
            "use_resonance_bias": getattr(config, "use_resonance_bias", None),
        },
        "before": before,
        "after": after,
        "history": history,
        "elapsed_sec": elapsed,
    }


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_condition.setdefault(run["condition"], []).append(run)
    summary = {}
    for condition, rows in by_condition.items():
        acc = [row["after"]["nearest_same_nf_acc"] for row in rows]
        gap = [row["after"]["pos_neg_gap"] for row in rows]
        before_acc = [row["before"]["nearest_same_nf_acc"] for row in rows]
        summary[condition] = {
            "n": len(rows),
            "params": rows[0]["params"],
            "before_acc_mean": statistics.fmean(before_acc),
            "after_acc_mean": statistics.fmean(acc),
            "after_acc_std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
            "after_gap_mean": statistics.fmean(gap),
        }
    return summary


def write_report(path: Path, args: argparse.Namespace, runs: list[dict[str, Any]]) -> None:
    summary = summarise(runs)
    lines = [
        "# Topology-Bearing Proof Equivalence Evaluation",
        "",
        "Positive pairs are deduplicated programs with the same beta-normal form.",
        f"Train groups: `{args.train_groups}`, val groups: `{args.val_groups}`, "
        f"programs/group target: `{args.programs_per_group}`, epochs: `{args.epochs}`.",
        "",
        "| Condition | Params | Before Acc | After Acc | After Acc Std | After Pos-Neg Gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, row in summary.items():
        lines.append(
            f"| {condition} | {row['params']:,} | {row['before_acc_mean']:.4f} | "
            f"{row['after_acc_mean']:.4f} | {row['after_acc_std']:.4f} | "
            f"{row['after_gap_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a topology-bearing task, unlike the generic local",
            "LM control. A resonance win here would support the architectural hypothesis;",
            "a loss here means the current phase/relation implementation still is not",
            "exploiting verified proof equivalence under these conditions.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/topology_eval"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", default="standard,resonance_full,bias_only,resonance_inert")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_groups", type=int, default=96)
    parser.add_argument("--val_groups", type=int, default=48)
    parser.add_argument("--eval_groups", type=int, default=40)
    parser.add_argument("--programs_per_group", type=int, default=6)
    parser.add_argument("--min_programs", type=int, default=2)
    parser.add_argument("--max_term_depth", type=int, default=5)
    parser.add_argument("--max_term_size", type=int, default=14)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--phase_init_std", type=float, default=0.3)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lm_weight", type=float, default=0.5)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--dataset_seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_deterministic(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    random.seed(args.dataset_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = NormalFormProofDataset(
        requested_groups=args.train_groups,
        programs_per_group=args.programs_per_group,
        min_programs=args.min_programs,
        max_term_depth=args.max_term_depth,
        max_term_size=args.max_term_size,
        seed=args.dataset_seed,
    )
    val_ds = NormalFormProofDataset(
        requested_groups=args.val_groups,
        programs_per_group=args.programs_per_group,
        min_programs=args.min_programs,
        max_term_depth=args.max_term_depth,
        max_term_size=args.max_term_size,
        seed=args.dataset_seed + 1,
    )
    print(f"Dataset: train_groups={len(train_ds)} val_groups={len(val_ds)}")
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    runs = []
    for seed in args.seeds:
        for condition in conditions:
            runs.append(train_condition(args, condition, seed, train_ds, val_ds))
    results = {
        "summary": summarise(runs),
        "runs": runs,
        "dataset": {
            "train_groups": len(train_ds),
            "val_groups": len(val_ds),
        },
        "config": vars(args),
    }
    (args.output_dir / "results.json").write_text(json.dumps(jsonify(results), indent=2))
    write_report(args.output_dir / "report.md", args, runs)
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
