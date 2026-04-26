#!/usr/bin/env python3
"""Reproducible local evaluation for the resonance-transformer idea.

This script deliberately avoids network datasets.  It runs small controlled
experiments that answer narrow questions:

1. Does the resonance model beat a standard transformer on a structured local
   language-modeling task under the same seed/data/hyperparameters?
2. Which component is doing useful work: phase stream, attention bias, or just
   extra parameters?
3. Are the synthetic proof-walk positives actually semantic positives under
   beta-normal-form equality?
4. Does the contrastive proof-walk objective learn an invariant representation
   on held-out proof groups?

The results are not intended as benchmark claims.  They are a falsifiable
smoke-test suite and a baseline for deeper GPU runs.
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
from torch.utils.data import DataLoader

from resonance.config import ResonanceConfig, StandardConfig
from resonance.device import enable_deterministic, get_device, set_seed
from resonance.models import ResonanceTransformer, StandardTransformer
from resonance.training import train_model
from train import StructuredSyntheticDataset

from synthetic.contrastive_trainer import (
    DualContrastiveLoss,
    MinimalResonanceTransformer,
    ProgramTokenizer,
)
from synthetic.proof_walk_generator import CrawlMode, ProofWalkDataset
from synthetic.lambda_generator import normal_form


def _jsonify(value: Any) -> Any:
    """Convert common torch/numpy-ish values to JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return value


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def config_param_summary(config: StandardConfig | ResonanceConfig) -> dict[str, Any]:
    return {
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "embed_dim": config.embed_dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "ff_dim": config.ff_dim,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "gradient_accumulation": config.gradient_accumulation,
        **(
            {
                "n_frequencies": config.n_frequencies,
                "use_phase_stream": config.use_phase_stream,
                "use_resonance_bias": config.use_resonance_bias,
                "phase_init_std": config.phase_init_std,
                "resonance_blend": config.resonance_blend,
                "resonance_attn_weight": config.resonance_attn_weight,
            }
            if isinstance(config, ResonanceConfig)
            else {}
        ),
    }


def build_lm_model(
    condition: str,
    *,
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    ff_dim: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    phase_init_std: float,
    wide_phase_init_std: float,
    resonance_attn_weight: float,
) -> tuple[torch.nn.Module, StandardConfig | ResonanceConfig]:
    wide_phase = condition.endswith("_wide")
    if wide_phase:
        condition = condition.removesuffix("_wide")

    if condition == "standard":
        config = StandardConfig(
            name="standard",
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation=1,
        )
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
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        n_frequencies=32,
        phase_init_std=wide_phase_init_std if wide_phase else phase_init_std,
        resonance_attn_weight=resonance_attn_weight,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation=1,
        phonetic_init=False,
        use_phase_stream=condition in {"resonance_full", "phase_stream_only"},
        use_resonance_bias=condition in {"resonance_full", "bias_only"},
    )
    return ResonanceTransformer(config), config


@torch.no_grad()
def evaluate_lm(
    model: torch.nn.Module,
    val_dataset: torch.utils.data.Dataset,
    config: StandardConfig | ResonanceConfig,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_seen = 0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        logits = out["logits"][:, :-1, :]
        targets = labels[:, 1:]
        valid = targets != -100
        total_loss += loss.item() * int(valid.sum().item())
        total_tokens += int(valid.sum().item())
        preds = logits.argmax(dim=-1)
        total_correct += int(((preds == targets) & valid).sum().item())
        total_seen += int(valid.sum().item())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "ppl": float(math.exp(min(avg_loss, 50.0))),
        "acc1": total_correct / max(total_seen, 1),
    }


def train_lm_condition(args: argparse.Namespace, condition: str, seed: int) -> dict[str, Any]:
    set_seed(seed)
    device = get_device(args.device)
    train_ds = StructuredSyntheticDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.train_samples,
        num_modes=args.num_modes,
        seed=seed,
    )
    val_ds = StructuredSyntheticDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.val_samples,
        num_modes=args.num_modes,
        seed=seed + 10_000,
    )
    model, config = build_lm_model(
        condition,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        phase_init_std=args.phase_init_std,
        wide_phase_init_std=args.wide_phase_init_std,
        resonance_attn_weight=args.resonance_attn_weight,
    )
    start = time.perf_counter()
    history = train_model(
        model,
        config,
        train_ds,
        val_ds,
        n_epochs=args.epochs,
        device=device,
        weight_decay=args.weight_decay,
        log_interval=max(1, len(train_ds) // args.batch_size + 1),
    )
    elapsed = time.perf_counter() - start
    model = model.to(device)
    clean = evaluate_lm(model, val_ds, config, device)
    interventions = {}
    if isinstance(model, ResonanceTransformer):
        interventions = resonance_interventions(model, val_ds, config, device)
    return {
        "condition": condition,
        "seed": seed,
        "params": count_parameters(model),
        "trainable_params": count_parameters(model, trainable_only=True),
        "config": config_param_summary(config),
        "history": history,
        "final_eval": clean,
        "interventions": interventions,
        "elapsed_sec": elapsed,
        "sec_per_epoch": elapsed / max(args.epochs, 1),
    }


def restore_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v.clone() for k, v in state.items()})


def perturb_parameter(param: torch.Tensor, sigma: float) -> None:
    std = param.std(unbiased=False).item()
    if not math.isfinite(std) or std == 0:
        std = 1.0
    param.add_(torch.randn_like(param) * sigma * std)


def resonance_interventions(
    model: ResonanceTransformer,
    val_ds: torch.utils.data.Dataset,
    config: ResonanceConfig,
    device: torch.device,
) -> dict[str, Any]:
    original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    original_phase_stream = config.use_phase_stream
    original_bias = config.use_resonance_bias

    def run(name: str) -> dict[str, float]:
        return evaluate_lm(model, val_ds, config, device, max_batches=20)

    results: dict[str, Any] = {"clean": run("clean")}

    config.use_resonance_bias = False
    results["disable_resonance_bias"] = run("disable_resonance_bias")
    config.use_resonance_bias = original_bias

    config.use_phase_stream = False
    results["semantic_only_disable_phase_stream"] = run("semantic_only")
    config.use_phase_stream = original_phase_stream

    with torch.no_grad():
        perm = torch.randperm(model.embedding.phase.weight.size(0), device=device)
        model.embedding.phase.weight.copy_(model.embedding.phase.weight[perm])
    results["permute_phase_embeddings"] = run("permute_phase_embeddings")
    restore_state(model, original_state)

    for sigma in (0.05, 0.20):
        with torch.no_grad():
            perturb_parameter(model.embedding.phase.weight, sigma)
        results[f"phase_noise_{sigma}"] = run(f"phase_noise_{sigma}")
        restore_state(model, original_state)

        with torch.no_grad():
            perturb_parameter(model.embedding.semantic.weight, sigma)
        results[f"semantic_noise_{sigma}"] = run(f"semantic_noise_{sigma}")
        restore_state(model, original_state)

    with torch.no_grad():
        phase = model.embedding.phase.weight.detach().cpu()
        centered = phase - phase.mean(dim=0)
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        var = (s * s) / (s * s).sum().clamp(min=1e-12)
        cum = torch.cumsum(var, dim=0)
        rank95 = int((cum < 0.95).sum().item() + 1)
        blend = torch.sigmoid(model.embedding.blend).detach().cpu()
        weights = torch.stack(
            [block.attn.resonance_weight.detach().cpu() for block in model.blocks]
        )
    results["phase_geometry"] = {
        "phase_rank95": rank95,
        "phase_var_top5": var[:5].tolist(),
        "blend_sigmoid_mean": float(blend.mean().item()),
        "blend_sigmoid_std": float(blend.std(unbiased=False).item()),
        "resonance_weight_mean": float(weights.mean().item()),
        "resonance_weight_std": float(weights.std(unbiased=False).item()),
    }

    config.use_phase_stream = original_phase_stream
    config.use_resonance_bias = original_bias
    restore_state(model, original_state)
    return results


def run_lm_suite(args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for seed in args.seeds:
        for condition in conditions:
            print(f"\n[LM] condition={condition} seed={seed}")
            results.append(train_lm_condition(args, condition, seed))
    return results


def audit_proof_groups(args: argparse.Namespace) -> dict[str, Any]:
    audits = {}
    for mode in (CrawlMode.STRUCTURED, CrawlMode.RANDOM, CrawlMode.HYBRID):
        dataset = ProofWalkDataset(
            n_groups=args.proof_audit_groups,
            programs_per_group=args.programs_per_group,
            max_term_depth=args.max_term_depth,
            max_term_size=args.max_term_size,
            generator_seed=args.proof_seed,
            crawl_mode=mode,
            structured_ratio=0.5,
        )
        same_nf_groups = 0
        unique_nf_counts = []
        duplicate_program_counts = []
        examples = []
        for i in range(len(dataset)):
            group = dataset.get_full_group(i)
            nfs = [normal_form(term).to_string() for term in group["program_terms"]]
            unique_nfs = sorted(set(nfs))
            unique_nf_counts.append(len(unique_nfs))
            duplicate_program_counts.append(
                len(group["programs"]) - len(set(group["programs"]))
            )
            if len(unique_nfs) == 1:
                same_nf_groups += 1
            elif len(examples) < 3:
                examples.append(
                    {
                        "statement": group["statement"],
                        "programs": group["programs"][:3],
                        "normal_forms": unique_nfs[:3],
                    }
                )
        audits[mode.name.lower()] = {
            "generated_groups": len(dataset),
            "same_normal_form_groups": same_nf_groups,
            "same_normal_form_rate": same_nf_groups / max(len(dataset), 1),
            "avg_unique_normal_forms": statistics.fmean(unique_nf_counts)
            if unique_nf_counts
            else 0.0,
            "avg_duplicate_programs_per_group": statistics.fmean(duplicate_program_counts)
            if duplicate_program_counts
            else 0.0,
            "counterexamples": examples,
        }
    return audits


@torch.no_grad()
def proof_retrieval_eval(
    model: MinimalResonanceTransformer,
    dataset: ProofWalkDataset,
    tokenizer: ProgramTokenizer,
    device: torch.device,
    max_groups: int,
) -> dict[str, float]:
    model.eval()
    all_programs: list[str] = []
    group_ids: list[int] = []
    for group_idx in range(min(max_groups, len(dataset))):
        item = dataset[group_idx]
        for program in item["programs"]:
            all_programs.append(program)
            group_ids.append(group_idx)
    tokens = tokenizer.batch_encode(all_programs, device=str(device))
    _, sem, phase = model(tokens)
    metrics = {}
    for name, emb in (("semantic", sem), ("phase", phase)):
        emb = F.normalize(emb, dim=-1)
        sim = emb @ emb.T
        sim.fill_diagonal_(-float("inf"))
        nearest = sim.argmax(dim=1).detach().cpu().tolist()
        group_tensor = torch.tensor(group_ids)
        correct = sum(1 for i, j in enumerate(nearest) if group_ids[i] == group_ids[j])
        pos_vals = []
        neg_vals = []
        for i in range(len(group_ids)):
            same = group_tensor == group_ids[i]
            same[i] = False
            diff = ~same
            diff[i] = False
            row = sim[i].detach().cpu()
            if same.any():
                pos_vals.append(float(row[same].mean().item()))
            if diff.any():
                neg_vals.append(float(row[diff].mean().item()))
        metrics[f"{name}_nearest_same_group_acc"] = correct / max(len(group_ids), 1)
        metrics[f"{name}_mean_positive_similarity"] = statistics.fmean(pos_vals)
        metrics[f"{name}_mean_negative_similarity"] = statistics.fmean(neg_vals)
        metrics[f"{name}_pos_neg_gap"] = (
            metrics[f"{name}_mean_positive_similarity"]
            - metrics[f"{name}_mean_negative_similarity"]
        )
    return metrics


def train_proof_contrast(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.proof_seed)
    device = get_device(args.device)
    train_ds = ProofWalkDataset(
        n_groups=args.proof_train_groups,
        programs_per_group=args.programs_per_group,
        max_term_depth=args.max_term_depth,
        max_term_size=args.max_term_size,
        generator_seed=args.proof_seed,
        crawl_mode=CrawlMode.HYBRID,
        structured_ratio=0.5,
    )
    val_ds = ProofWalkDataset(
        n_groups=args.proof_val_groups,
        programs_per_group=args.programs_per_group,
        max_term_depth=args.max_term_depth,
        max_term_size=args.max_term_size,
        generator_seed=args.proof_seed + 1,
        crawl_mode=CrawlMode.HYBRID,
        structured_ratio=0.5,
    )
    tokenizer = ProgramTokenizer(max_length=args.program_max_length)
    model = MinimalResonanceTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.proof_d_model,
        n_layers=args.proof_layers,
        n_heads=args.proof_heads,
        d_semantic=args.proof_d_model // 2,
        d_phase=args.proof_d_model // 2,
        max_length=args.program_max_length,
        dropout=args.dropout,
    ).to(device)
    before = proof_retrieval_eval(
        model, val_ds, tokenizer, device, max_groups=args.proof_eval_groups
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.proof_batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.proof_learning_rate)
    contrastive = DualContrastiveLoss(temperature=args.temperature)
    history = {"lm_loss": [], "contrastive_loss": [], "total_loss": []}
    start = time.perf_counter()
    for epoch in range(args.proof_epochs):
        model.train()
        epoch_lm = 0.0
        epoch_ctr = 0.0
        epoch_total = 0.0
        n_batches = 0
        for batch in loader:
            all_programs = []
            group_sizes = []
            for group in batch["programs"]:
                all_programs.extend(group)
                group_sizes.append(len(group))
            tokens = tokenizer.batch_encode(all_programs, device=str(device))
            logits, sem, phase = model(tokens)
            lm_loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
                tokens[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id,
            )
            total = sum(group_sizes)
            positive_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            offset = 0
            for size in group_sizes:
                positive_mask[offset : offset + size, offset : offset + size] = True
                offset += size
            diag = torch.arange(total, device=device)
            positive_mask[diag, diag] = False
            negative_mask = ~positive_mask
            negative_mask[diag, diag] = False
            ctr_loss, _ = contrastive(sem, phase, positive_mask, negative_mask)
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
            f"[Proof] epoch={epoch + 1} "
            f"lm={history['lm_loss'][-1]:.4f} "
            f"ctr={history['contrastive_loss'][-1]:.4f}"
        )
    elapsed = time.perf_counter() - start
    after = proof_retrieval_eval(
        model, val_ds, tokenizer, device, max_groups=args.proof_eval_groups
    )
    return {
        "params": count_parameters(model),
        "train_groups_generated": len(train_ds),
        "val_groups_generated": len(val_ds),
        "history": history,
        "before": before,
        "after": after,
        "elapsed_sec": elapsed,
    }


def summarise_lm(lm_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in lm_results:
        by_condition.setdefault(row["condition"], []).append(row)
    summary = {}
    for condition, rows in by_condition.items():
        ppls = [r["final_eval"]["ppl"] for r in rows]
        accs = [r["final_eval"]["acc1"] for r in rows]
        summary[condition] = {
            "n": len(rows),
            "ppl_mean": statistics.fmean(ppls),
            "ppl_std": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            "acc1_mean": statistics.fmean(accs),
            "params": rows[0]["params"],
            "sec_per_epoch_mean": statistics.fmean(r["sec_per_epoch"] for r in rows),
        }
    if "standard" in summary:
        base = summary["standard"]["ppl_mean"]
        for condition in summary:
            summary[condition]["ppl_ratio_vs_standard"] = (
                summary[condition]["ppl_mean"] / base
            )
    return summary


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    lm_results: list[dict[str, Any]],
    proof_audit: dict[str, Any],
    proof_contrast: dict[str, Any],
) -> Path:
    summary = summarise_lm(lm_results)
    path = out_dir / "report.md"
    best = min(summary.items(), key=lambda kv: kv[1]["ppl_mean"])[0] if summary else "n/a"
    lines = [
        "# Resonance Evaluation Report",
        "",
        "## Scope",
        "",
        "This is a local, reproducible smoke suite. It does not use TinyStories,",
        "internet datasets, GPUs, or external benchmark comparisons. Treat the",
        "numbers as evidence about this implementation under controlled synthetic",
        "tasks, not as a publication claim.",
        "",
        "## Language-Modeling Ablation",
        "",
        f"Seeds: `{list(args.seeds)}`. Dataset: structured synthetic, "
        f"`{args.train_samples}` train sequences, `{args.val_samples}` validation "
        f"sequences, seq_len `{args.seq_len}`, vocab `{args.vocab_size}`.",
        "",
        "| Condition | Params | Mean PPL | PPL Std | Mean Acc@1 | PPL Ratio vs Standard | Sec/Epoch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in summary.items():
        lines.append(
            f"| {condition} | {row['params']:,} | {row['ppl_mean']:.3f} | "
            f"{row['ppl_std']:.3f} | {row['acc1_mean']:.4f} | "
            f"{row.get('ppl_ratio_vs_standard', 1.0):.3f} | "
            f"{row['sec_per_epoch_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Best mean PPL in this run: `{best}`.",
            "",
            "Interpretation: the only defensible claim from this table is whether the",
            "phase stream / resonance bias improves this exact synthetic objective.",
            "The standard control, phase-stream-only control, bias-only control,",
            "and inert-resonance control are required because the full model has",
            "extra parameters.",
            "",
            "## Resonance Interventions",
            "",
            "Each resonance condition also records inference-time interventions in",
            "`results.json`: disabling the attention bias, disabling the phase stream,",
            "permuting phase embeddings, and adding phase/semantic noise. Large PPL",
            "deltas under phase-specific interventions would support the claim that",
            "the learned phase subsystem is behaviorally used; small deltas would make",
            "the wave-language mostly decorative for this task.",
            "",
            "## Proof-Walk Audit",
            "",
            "| Crawl Mode | Generated Groups | Same Normal-Form Rate | Avg Unique NFs | Avg Duplicate Programs |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode, row in proof_audit.items():
        lines.append(
            f"| {mode} | {row['generated_groups']} | "
            f"{100 * row['same_normal_form_rate']:.1f}% | "
            f"{row['avg_unique_normal_forms']:.3f} | "
            f"{row['avg_duplicate_programs_per_group']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The proof generator is close to producing semantic-positive groups under",
            "beta-normal-form equality, but duplicate programs are common. For serious",
            "contrastive training, positives should be deduplicated and counterexample",
            "groups should be either repaired or excluded.",
            "",
            "## Contrastive Proof-Walk Retrieval",
            "",
            "| Metric | Before | After |",
            "|---|---:|---:|",
        ]
    )
    for key in sorted(proof_contrast["before"]):
        before = proof_contrast["before"][key]
        after = proof_contrast["after"][key]
        lines.append(f"| {key} | {before:.4f} | {after:.4f} |")
    lines.extend(
        [
            "",
            "This checks whether the contrastive objective learns an invariant program",
            "representation on held-out generated proof groups. It does not yet show",
            "transfer to natural language or real code.",
            "",
            "## What's Real",
            "",
            "- The concrete architectural idea is a learnable pairwise attention bias: "
            "`R[i,j] = mean_f cos(phi[token_i,f] - phi[token_j,f])`, plus an optional "
            "low-dimensional phase stream mixed into token embeddings.",
            "- This is in the family of relation-aware self-attention and structural",
            "attention biasing. The potentially interesting angle is not novelty of",
            "attention bias, but whether a compact learned phase code is a good carrier",
            "for semantic-preserving equivalence classes.",
            "- The proof-walk and semantic-preserving mutation idea is real and worth",
            "pursuing, but it needs a stricter semantics oracle than same type/context.",
            "",
            "## What Next",
            "",
            "1. Replace type-only positives with normal-form-verified positives and",
            "   deduplicate each proof group.",
            "2. Add iso-parameter baselines. The resonance model currently has extra",
            "   parameters, so a same-width standard baseline is not enough.",
            "3. Run multi-seed standardized text experiments only after the local",
            "   ablations show a stable effect.",
            "4. For real code, use compiler/property-test-backed semantic-preserving",
            "   transforms and evaluate clone retrieval, perturbation stability, and",
            "   transfer to code summarization/type tasks.",
            "",
            "## Quantum / Microtubule Material",
            "",
            "For this project, treat it as flavor or analogy unless a concrete operator",
            "falls out of it. Superradiance/interference language can motivate low-rank",
            "collective coupling or phase kernels, but it should not appear as evidence",
            "for the ML claims. The ML work stands or falls on controlled baselines,",
            "ablation, semantics-preserving data, and scaling behavior.",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/research_eval"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument(
        "--conditions",
        default="standard,resonance_full,phase_stream_only,bias_only,resonance_inert",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_samples", type=int, default=768)
    parser.add_argument("--val_samples", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--num_modes", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--phase_init_std", type=float, default=0.3)
    parser.add_argument("--wide_phase_init_std", type=float, default=math.pi)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)

    parser.add_argument("--proof_seed", type=int, default=123)
    parser.add_argument("--proof_audit_groups", type=int, default=80)
    parser.add_argument("--proof_train_groups", type=int, default=64)
    parser.add_argument("--proof_val_groups", type=int, default=32)
    parser.add_argument("--proof_eval_groups", type=int, default=24)
    parser.add_argument("--programs_per_group", type=int, default=6)
    parser.add_argument("--max_term_depth", type=int, default=4)
    parser.add_argument("--max_term_size", type=int, default=10)
    parser.add_argument("--program_max_length", type=int, default=96)
    parser.add_argument("--proof_d_model", type=int, default=64)
    parser.add_argument("--proof_layers", type=int, default=2)
    parser.add_argument("--proof_heads", type=int, default=2)
    parser.add_argument("--proof_batch_size", type=int, default=8)
    parser.add_argument("--proof_epochs", type=int, default=3)
    parser.add_argument("--proof_learning_rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--skip_lm", action="store_true")
    parser.add_argument("--skip_proof_contrast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    enable_deterministic(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w") as f:
        json.dump(_jsonify(vars(args)), f, indent=2)

    lm_results = [] if args.skip_lm else run_lm_suite(args)
    proof_audit = audit_proof_groups(args)
    proof_contrast = (
        {"before": {}, "after": {}, "history": {}}
        if args.skip_proof_contrast
        else train_proof_contrast(args)
    )

    results = {
        "lm_summary": summarise_lm(lm_results),
        "lm_runs": lm_results,
        "proof_audit": proof_audit,
        "proof_contrast": proof_contrast,
    }
    with (args.output_dir / "results.json").open("w") as f:
        json.dump(_jsonify(results), f, indent=2)
    report = write_report(args.output_dir, args, lm_results, proof_audit, proof_contrast)
    print(f"\nWrote results: {args.output_dir / 'results.json'}")
    print(f"Wrote report:  {report}")


if __name__ == "__main__":
    main()
