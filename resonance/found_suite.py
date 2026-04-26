#!/usr/bin/env python3
"""Foundational Experiment Suite for the Resonance Transformer.

Six experiments testing the core hypothesis:
    "Can a compact learned phase code represent semantic-preserving
     equivalence structure more efficiently than ordinary embeddings?"

Experiments:
    1. ablation      — Multi-seed iso-parameter ablation (standard vs resonance variants)
    2. geometry      — Phase embedding geometry & compressibility analysis
    3. perturbation  — Perturbation intervention battery
    4. retrieval     — Equivalence retrieval benchmark on proof-walks
    5. bias_analysis — Attention bias function analysis
    6. iso_param     — Iso-parameter architecture search

Usage:
    python found_suite.py --experiment ablation --seeds 11 23 42 --epochs 10
    python found_suite.py --experiment geometry --checkpoint path/to/checkpoint.pt
    python found_suite.py --all --output_dir outputs/foundational
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure package imports work when run from repo root or resonance/ directory
sys.path.insert(0, str(Path(__file__).parent))

from resonance.config import ResonanceConfig, StandardConfig
from resonance.device import enable_deterministic, get_device, set_seed
from resonance.models import ResonanceTransformer, StandardTransformer
from resonance.training import train_model
from train import StructuredSyntheticDataset

from foundational.iso_param import build_iso_pair, print_param_table


# =============================================================================
# Shared Utilities
# =============================================================================

def _jsonify(value: Any) -> Any:
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate_lm(
    model: nn.Module,
    dataset: Dataset,
    config: StandardConfig | ResonanceConfig,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        logits = out["logits"][:, :-1, :]
        targets = labels[:, 1:]
        valid = targets != -100
        ntok = int(valid.sum().item())
        total_loss += loss.item() * ntok
        total_tokens += ntok
        preds = logits.argmax(dim=-1)
        total_correct += int(((preds == targets) & valid).sum().item())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "ppl": float(math.exp(min(avg_loss, 50.0))),
        "acc1": total_correct / max(total_tokens, 1),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: StandardConfig | ResonanceConfig,
    history: dict[str, list[float]],
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "history": history,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, StandardConfig | ResonanceConfig, dict[str, Any]]:
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, ResonanceConfig):
        model = ResonanceTransformer(config)
    else:
        model = StandardTransformer(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    return model, config, ckpt.get("metadata", {})


# =============================================================================
# Experiment 1: Multi-Seed Ablation with Iso-Parameter Matching
# =============================================================================

def run_ablation(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Run multi-seed ablation comparing standard and resonance variants."""
    device = get_device(args.device)
    results: dict[str, list[dict[str, Any]]] = {}

    conditions = {
        "standard": {"type": "standard"},
        "resonance_full": {"type": "resonance", "phase": True, "bias": True},
        "phase_only": {"type": "resonance", "phase": True, "bias": False},
        "bias_only": {"type": "resonance", "phase": False, "bias": True},
        "inert": {"type": "resonance", "phase": False, "bias": False},
    }

    for seed in args.seeds:
        for name, cfg in conditions.items():
            set_seed(seed)
            print(f"\n{'='*60}")
            print(f"Ablation: {name} | Seed: {seed}")
            print(f"{'='*60}")

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

            if cfg["type"] == "standard":
                # Build iso-parameter standard model
                _, model, std_config, res_config = build_iso_pair(
                    base_embed_dim=args.embed_dim,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    ff_dim=args.ff_dim,
                    vocab_size=args.vocab_size,
                    seq_len=args.seq_len,
                    n_frequencies=args.n_frequencies,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    gradient_accumulation=args.gradient_accumulation,
                )
                config = std_config
            else:
                # Build resonance with requested config
                config = ResonanceConfig(
                    name=name,
                    vocab_size=args.vocab_size,
                    max_seq_len=args.seq_len,
                    embed_dim=args.embed_dim,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    ff_dim=args.ff_dim,
                    n_frequencies=args.n_frequencies,
                    resonance_blend=args.resonance_blend,
                    resonance_attn_weight=args.resonance_attn_weight,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    gradient_accumulation=args.gradient_accumulation,
                    learning_rate=args.learning_rate,
                    phase_init_std=args.phase_init_std,
                    phonetic_init=False,
                    use_phase_stream=cfg.get("phase", True),
                    use_resonance_bias=cfg.get("bias", True),
                )
                model = ResonanceTransformer(config)

            print(f"  Params: {count_params(model):,}")

            start = time.perf_counter()
            history = train_model(
                model, config, train_ds, val_ds,
                n_epochs=args.epochs,
                device=device,
                weight_decay=args.weight_decay,
                log_interval=max(1, len(train_ds) // args.batch_size + 1),
            )
            elapsed = time.perf_counter() - start

            final_eval = evaluate_lm(model, val_ds, config, device)

            ckpt_dir = output_dir / name / f"seed{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_dir / "latest.pt",
                model, config, history,
                metadata={"seed": seed, "condition": name, "elapsed_sec": elapsed},
            )

            run_result = {
                "condition": name,
                "seed": seed,
                "params": count_params(model),
                "history": history,
                "final_eval": final_eval,
                "elapsed_sec": elapsed,
            }
            results.setdefault(name, []).append(run_result)

    # Summarize
    summary = {}
    for name, runs in results.items():
        ppls = [r["final_eval"]["ppl"] for r in runs]
        accs = [r["final_eval"]["acc1"] for r in runs]
        summary[name] = {
            "n": len(runs),
            "ppl_mean": statistics.fmean(ppls),
            "ppl_std": statistics.stdev(ppls) if len(ppls) > 1 else 0.0,
            "acc1_mean": statistics.fmean(accs),
            "params": runs[0]["params"],
        }

    if "standard" in summary:
        base = summary["standard"]["ppl_mean"]
        for name in summary:
            summary[name]["ppl_ratio_vs_standard"] = summary[name]["ppl_mean"] / base

    # Effect sizes
    if "standard" in results and "resonance_full" in results:
        std_ppls = [r["final_eval"]["ppl"] for r in results["standard"]]
        res_ppls = [r["final_eval"]["ppl"] for r in results["resonance_full"]]
        if len(std_ppls) > 1 and len(res_ppls) > 1:
            # Paired Cohen's d
            diffs = [s - r for s, r in zip(std_ppls, res_ppls)]
            d_mean = statistics.fmean(diffs)
            d_std = statistics.stdev(diffs) if len(diffs) > 1 else 1.0
            summary["effect_size"] = {
                "cohens_d": d_mean / max(d_std, 1e-12),
                "mean_diff": d_mean,
            }

    return {"summary": summary, "runs": results}


# =============================================================================
# Experiment 2: Phase Geometry & Compressibility
# =============================================================================

def run_geometry(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Analyze phase embedding geometry on a trained resonance model."""
    device = get_device(args.device)

    if args.checkpoint:
        model, config, _ = load_checkpoint(Path(args.checkpoint), device)
        if not isinstance(model, ResonanceTransformer):
            raise ValueError("Geometry analysis requires a ResonanceTransformer checkpoint")
    else:
        # Train a small model for analysis
        print("No checkpoint provided, training a small model for geometry analysis...")
        set_seed(42)
        config = ResonanceConfig(
            name="geometry",
            vocab_size=args.vocab_size,
            max_seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ff_dim=args.ff_dim,
            n_frequencies=args.n_frequencies,
            dropout=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        model = ResonanceTransformer(config).to(device)
        train_ds = StructuredSyntheticDataset(
            vocab_size=args.vocab_size, seq_len=args.seq_len,
            num_samples=args.train_samples, num_modes=args.num_modes, seed=42,
        )
        val_ds = StructuredSyntheticDataset(
            vocab_size=args.vocab_size, seq_len=args.seq_len,
            num_samples=args.val_samples, num_modes=args.num_modes, seed=42+10_000,
        )
        train_model(model, config, train_ds, val_ds, n_epochs=args.epochs, device=device)

    model.eval()

    # Extract embeddings
    sem_weight = model.embedding.semantic.weight.detach().cpu()  # [V, D]
    phase_weight = model.embedding.phase.weight.detach().cpu()   # [V, F]
    blend = torch.sigmoid(model.embedding.blend).detach().cpu()

    # Effective rank analysis
    def effective_rank(matrix: torch.Tensor, threshold: float = 0.95) -> dict[str, Any]:
        centered = matrix - matrix.mean(dim=0)
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        var = (s * s) / (s * s).sum().clamp(min=1e-12)
        cumvar = torch.cumsum(var, dim=0)
        rank95 = int((cumvar < threshold).sum().item() + 1)
        rank99 = int((cumvar < 0.99).sum().item() + 1)
        return {
            "effective_rank_95": rank95,
            "effective_rank_99": rank99,
            "total_dims": matrix.size(1),
            "rank_ratio_95": rank95 / matrix.size(1),
            "singular_values_top10": s[:10].tolist(),
            "variance_explained_top5": var[:5].tolist(),
        }

    sem_geom = effective_rank(sem_weight)
    phase_geom = effective_rank(phase_weight)

    # Cross-predictability: Ridge regression implemented manually
    def ridge_regression(X: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Return predictions from Ridge regression: beta = (X^T X + alpha*I)^-1 X^T y."""
        X_b = torch.cat([X, torch.ones(X.size(0), 1)], dim=1)  # add bias
        XtX = X_b.T @ X_b
        XtX.diagonal()[:-1].add_(alpha)  # don't regularize bias
        beta = torch.linalg.solve(XtX, X_b.T @ y)
        return X_b @ beta

    def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum()
        return float((1 - ss_res / ss_tot.clamp(min=1e-12)).item())

    # Ridge: phase -> semantic
    sem_pred_from_phase = ridge_regression(phase_weight, sem_weight, alpha=1.0)
    r2_phase_to_semantic = r2_score_torch(sem_weight, sem_pred_from_phase)

    # Ridge: semantic -> phase
    phase_pred_from_sem = ridge_regression(sem_weight, phase_weight, alpha=1.0)
    r2_semantic_to_phase = r2_score_torch(phase_weight, phase_pred_from_sem)

    # Random baseline: predict semantic from random projection
    random_proj = torch.randn_like(phase_weight)
    sem_pred_from_rand = ridge_regression(random_proj, sem_weight, alpha=1.0)
    r2_random_to_semantic = r2_score_torch(sem_weight, sem_pred_from_rand)

    # Blend statistics
    blend_stats = {
        "mean": float(blend.mean().item()),
        "std": float(blend.std(unbiased=False).item()),
        "median": float(blend.median().item()),
        "min": float(blend.min().item()),
        "max": float(blend.max().item()),
        "fraction_above_0.5": float((blend > 0.5).float().mean().item()),
    }

    # Resonance weight statistics
    res_weights = torch.stack([
        block.attn.resonance_weight.detach().cpu()
        for block in model.blocks
    ])
    res_weight_stats = {
        "mean": float(res_weights.mean().item()),
        "std": float(res_weights.std(unbiased=False).item()),
        "per_layer": res_weights.tolist(),
    }

    results = {
        "semantic_geometry": sem_geom,
        "phase_geometry": phase_geom,
        "cross_predictability": {
            "r2_phase_to_semantic": float(r2_phase_to_semantic),
            "r2_semantic_to_phase": float(r2_semantic_to_phase),
            "r2_random_to_semantic": float(r2_random_to_semantic),
            "phase_better_than_random": float(r2_phase_to_semantic) > float(r2_random_to_semantic),
        },
        "blend_stats": blend_stats,
        "resonance_weight_stats": res_weight_stats,
    }

    print("\n" + "=" * 60)
    print("Phase Geometry Analysis")
    print("=" * 60)
    print(f"Semantic rank-95:  {sem_geom['effective_rank_95']} / {sem_geom['total_dims']} "
          f"({sem_geom['rank_ratio_95']:.2%})")
    print(f"Phase rank-95:     {phase_geom['effective_rank_95']} / {phase_geom['total_dims']} "
          f"({phase_geom['rank_ratio_95']:.2%})")
    print(f"R² (phase→semantic): {r2_phase_to_semantic:.4f}")
    print(f"R² (semantic→phase): {r2_semantic_to_phase:.4f}")
    print(f"R² (random→semantic): {r2_random_to_semantic:.4f}")
    print(f"Blend mean: {blend_stats['mean']:.3f}")
    print(f"Resonance weight mean: {res_weight_stats['mean']:.4f}")
    print("=" * 60)

    return results


# =============================================================================
# Experiment 3: Perturbation Intervention Battery
# =============================================================================

def run_perturbation(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Apply controlled perturbations to trained models and measure PPL deltas."""
    device = get_device(args.device)

    # We need checkpoints from the ablation experiment
    ckpt_dir = output_dir.parent / "ablation" if args.checkpoint_dir is None else Path(args.checkpoint_dir)

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}. "
            f"Run ablation experiment first or specify --checkpoint_dir"
        )

    # Load validation data
    val_ds = StructuredSyntheticDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.val_samples,
        num_modes=args.num_modes,
        seed=42 + 10_000,
    )

    def perturb(model: nn.Module, config: Any, kind: str, sigma: float | None = None) -> dict[str, float]:
        """Apply a perturbation and return eval metrics."""
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        if kind == "zero_phase":
            if hasattr(model, "embedding") and hasattr(model.embedding, "phase"):
                model.embedding.phase.weight.zero_()
        elif kind == "permute_phase":
            if hasattr(model, "embedding") and hasattr(model.embedding, "phase"):
                perm = torch.randperm(model.embedding.phase.weight.size(0), device=device)
                model.embedding.phase.weight.copy_(model.embedding.phase.weight[perm])
        elif kind == "noise_phase" and sigma is not None:
            if hasattr(model, "embedding") and hasattr(model.embedding, "phase"):
                std = model.embedding.phase.weight.std(unbiased=False).item()
                if std > 0:
                    model.embedding.phase.weight.add_(torch.randn_like(model.embedding.phase.weight) * sigma * std)
        elif kind == "zero_semantic":
            if hasattr(model, "embedding") and hasattr(model.embedding, "semantic"):
                model.embedding.semantic.weight.zero_()
            elif hasattr(model, "token_embed"):
                model.token_embed.weight.zero_()
        elif kind == "permute_semantic":
            if hasattr(model, "embedding") and hasattr(model.embedding, "semantic"):
                perm = torch.randperm(model.embedding.semantic.weight.size(0), device=device)
                model.embedding.semantic.weight.copy_(model.embedding.semantic.weight[perm])
            elif hasattr(model, "token_embed"):
                perm = torch.randperm(model.token_embed.weight.size(0), device=device)
                model.token_embed.weight.copy_(model.token_embed.weight[perm])
        elif kind == "noise_semantic" and sigma is not None:
            emb = model.embedding.semantic.weight if hasattr(model, "embedding") else model.token_embed.weight
            std = emb.std(unbiased=False).item()
            if std > 0:
                emb.add_(torch.randn_like(emb) * sigma * std)
        elif kind == "disable_bias":
            if hasattr(config, "use_resonance_bias"):
                config.use_resonance_bias = False
        elif kind == "disable_phase_stream":
            if hasattr(config, "use_phase_stream"):
                config.use_phase_stream = False

        result = evaluate_lm(model, val_ds, config, device, max_batches=50)

        # Restore
        if kind in ("disable_bias", "disable_phase_stream"):
            if hasattr(config, "use_resonance_bias"):
                config.use_resonance_bias = True
            if hasattr(config, "use_phase_stream"):
                config.use_phase_stream = True
        else:
            model.load_state_dict(original_state)

        return result

    all_results = {}

    for condition in ["standard", "resonance_full"]:
        condition_dir = ckpt_dir / condition
        if not condition_dir.exists():
            continue

        # Find first seed checkpoint
        seed_dirs = sorted(condition_dir.glob("seed*"))
        if not seed_dirs:
            continue

        ckpt_path = seed_dirs[0] / "latest.pt"
        if not ckpt_path.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Perturbation: {condition} (from {ckpt_path})")
        print(f"{'='*60}")

        model, config, _ = load_checkpoint(ckpt_path, device)
        clean = evaluate_lm(model, val_ds, config, device, max_batches=50)
        print(f"  Clean PPL: {clean['ppl']:.2f}")

        perturbations = [
            ("zero_semantic", None),
            ("permute_semantic", None),
            ("noise_semantic", 0.05),
            ("noise_semantic", 0.20),
        ]
        if isinstance(model, ResonanceTransformer):
            perturbations.extend([
                ("zero_phase", None),
                ("permute_phase", None),
                ("noise_phase", 0.05),
                ("noise_phase", 0.20),
                ("disable_bias", None),
                ("disable_phase_stream", None),
            ])

        condition_results = {"clean": clean}
        for kind, sigma in perturbations:
            name = f"{kind}_{sigma}" if sigma is not None else kind
            result = perturb(model, config, kind, sigma)
            delta = result["ppl"] - clean["ppl"]
            condition_results[name] = {**result, "delta_ppl": delta}
            print(f"  {name:<25} PPL: {result['ppl']:.2f}  Δ: {delta:+.2f}")

        all_results[condition] = condition_results

    # Compute phase-specificity score
    if "resonance_full" in all_results and "standard" in all_results:
        res = all_results["resonance_full"]
        std = all_results["standard"]
        # Phase-specificity: does phase perturbation hurt resonance more than standard?
        phase_specificity = {}
        for ptype in ["zero_phase", "permute_phase", "noise_phase_0.05", "noise_phase_0.20"]:
            if ptype in res:
                std_equiv = ptype.replace("phase", "semantic")
                if std_equiv in std:
                    phase_specificity[ptype] = {
                        "resonance_delta": res[ptype]["delta_ppl"],
                        "standard_delta": std[std_equiv]["delta_ppl"],
                        "specificity_ratio": res[ptype]["delta_ppl"] / max(std[std_equiv]["delta_ppl"], 0.01),
                    }
        all_results["phase_specificity"] = phase_specificity

    return all_results


# =============================================================================
# Experiment 4: Equivalence Retrieval Benchmark
# =============================================================================

def run_retrieval(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Train encoders on proof-walk equivalence and evaluate retrieval."""
    try:
        from synthetic.proof_walk_generator import ProofWalkDataset, CrawlMode
        from synthetic.lambda_generator import normal_form
        from synthetic.contrastive_trainer import (
            MinimalResonanceTransformer, ProgramTokenizer, DualContrastiveLoss
        )
    except ImportError as e:
        print(f"Synthetic pipeline not available: {e}")
        return {"error": str(e)}

    device = get_device(args.device)

    # Generate normal-form-verified proof groups
    print("\nGenerating normal-form-verified proof-walk groups...")
    train_groups = []
    val_groups = []
    rng = random.Random(args.proof_seed)

    def generate_verified_groups(n_groups: int, programs_per_group: int, seed: int) -> list[dict]:
        """Generate proof groups where all programs share the same normal form."""
        dataset = ProofWalkDataset(
            n_groups=n_groups * 3,  # oversample to account for filtering
            programs_per_group=programs_per_group,
            max_term_depth=args.max_term_depth,
            max_term_size=args.max_term_size,
            generator_seed=seed,
            crawl_mode=CrawlMode.HYBRID,
            structured_ratio=0.5,
        )
        verified = []
        for i in range(len(dataset)):
            group = dataset.get_full_group(i)
            programs = group["programs"]
            terms = group["program_terms"]
            # Compute normal forms
            nfs = []
            for term in terms:
                try:
                    nf = normal_form(term)
                    nfs.append(nf.to_string() if hasattr(nf, "to_string") else str(nf))
                except Exception:
                    nfs.append(None)

            # Keep only programs with the same normal form
            valid_nf = [nf for nf in nfs if nf is not None]
            if len(valid_nf) == 0:
                continue
            most_common = max(set(valid_nf), key=valid_nf.count)
            indices = [j for j, nf in enumerate(nfs) if nf == most_common]

            # Deduplicate
            seen = set()
            deduped_indices = []
            for j in indices:
                if programs[j] not in seen:
                    seen.add(programs[j])
                    deduped_indices.append(j)

            if len(deduped_indices) >= 2:
                verified.append({
                    "programs": [programs[j] for j in deduped_indices],
                    "nf": most_common,
                })
            if len(verified) >= n_groups:
                break
        return verified

    train_groups = generate_verified_groups(args.proof_train_groups, args.programs_per_group, args.proof_seed)
    val_groups = generate_verified_groups(args.proof_val_groups, args.programs_per_group, args.proof_seed + 1)

    print(f"  Verified train groups: {len(train_groups)}")
    print(f"  Verified val groups: {len(val_groups)}")

    tokenizer = ProgramTokenizer(max_length=args.program_max_length)

    # Build models
    d_model = args.proof_d_model
    conditions = {
        "standard_encoder": lambda: MinimalResonanceTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            n_layers=args.proof_layers,
            n_heads=args.proof_heads,
            d_semantic=d_model,
            d_phase=0,  # no phase
            max_length=args.program_max_length,
            dropout=args.dropout,
        ),
        "dual_head": lambda: MinimalResonanceTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            n_layers=args.proof_layers,
            n_heads=args.proof_heads,
            d_semantic=d_model // 2,
            d_phase=d_model // 2,
            max_length=args.program_max_length,
            dropout=args.dropout,
        ),
        "resonance_full": lambda: MinimalResonanceTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            n_layers=args.proof_layers,
            n_heads=args.proof_heads,
            d_semantic=d_model // 2,
            d_phase=d_model // 2,
            use_resonance_bias=True,
            max_length=args.program_max_length,
            dropout=args.dropout,
        ),
    }

    all_results = {}

    for name, model_fn in conditions.items():
        print(f"\n{'='*60}")
        print(f"Retrieval: {name}")
        print(f"{'='*60}")

        set_seed(args.proof_seed)
        model = model_fn().to(device)
        print(f"  Params: {count_params(model):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.proof_learning_rate)
        contrastive = DualContrastiveLoss(temperature=args.temperature)

        # Training
        for epoch in range(args.proof_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch by sampling groups
            batch_size = args.proof_batch_size
            random.shuffle(train_groups)
            for i in range(0, len(train_groups), batch_size):
                batch_groups = train_groups[i:i + batch_size]
                all_programs = []
                group_sizes = []
                for g in batch_groups:
                    all_programs.extend(g["programs"])
                    group_sizes.append(len(g["programs"]))

                tokens = tokenizer.batch_encode(all_programs, device=str(device))
                logits, sem, phase = model(tokens)

                # LM loss
                lm_loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, tokenizer.vocab_size),
                    tokens[:, 1:].reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )

                # Contrastive loss on semantic head
                total = sum(group_sizes)
                positive_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
                offset = 0
                for size in group_sizes:
                    positive_mask[offset:offset + size, offset:offset + size] = True
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

                epoch_loss += total_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch + 1}/{args.proof_epochs}: loss={avg_loss:.4f}")

        # Evaluation: retrieval on val groups
        model.eval()
        all_programs = []
        group_ids = []
        for gidx, group in enumerate(val_groups):
            for prog in group["programs"]:
                all_programs.append(prog)
                group_ids.append(gidx)

        tokens = tokenizer.batch_encode(all_programs, device=str(device))
        with torch.no_grad():
            _, sem, phase = model(tokens)

        results = {}
        for emb_name, emb in [("semantic", sem), ("phase", phase)]:
            emb = F.normalize(emb, dim=-1)
            sim = emb @ emb.T
            sim.fill_diagonal_(-float("inf"))

            # Recall@k
            for k in [1, 3, 5]:
                _, topk_indices = sim.topk(k, dim=-1)
                correct = 0
                total_q = 0
                for i in range(len(group_ids)):
                    same_group = [j for j in range(len(group_ids)) if group_ids[j] == group_ids[i] and j != i]
                    if len(same_group) == 0:
                        continue
                    retrieved = topk_indices[i, :k].cpu().tolist()
                    hits = len(set(retrieved) & set(same_group))
                    correct += hits / len(same_group)
                    total_q += 1
                results[f"{emb_name}_recall@{k}"] = correct / max(total_q, 1)

            # Positive-negative gap
            group_tensor = torch.tensor(group_ids)
            pos_vals = []
            neg_vals = []
            for i in range(len(group_ids)):
                same = group_tensor == group_ids[i]
                same[i] = False
                diff = ~same
                diff[i] = False
                if same.any():
                    pos_vals.append(float(sim[i][same].mean().item()))
                if diff.any():
                    neg_vals.append(float(sim[i][diff].mean().item()))
            results[f"{emb_name}_pos_mean"] = statistics.fmean(pos_vals) if pos_vals else 0.0
            results[f"{emb_name}_neg_mean"] = statistics.fmean(neg_vals) if neg_vals else 0.0
            results[f"{emb_name}_gap"] = results[f"{emb_name}_pos_mean"] - results[f"{emb_name}_neg_mean"]

        print(f"  Semantic recall@1: {results['semantic_recall@1']:.4f}")
        print(f"  Phase recall@1: {results['phase_recall@1']:.4f}")
        print(f"  Semantic gap: {results['semantic_gap']:.4f}")
        print(f"  Phase gap: {results['phase_gap']:.4f}")

        all_results[name] = results

    return all_results


# =============================================================================
# Experiment 5: Attention Bias Function Analysis
# =============================================================================

def run_bias_analysis(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Analyze what the resonance bias actually does in attention."""
    device = get_device(args.device)

    if not args.checkpoint:
        raise ValueError("--checkpoint required for bias_analysis")

    model, config, _ = load_checkpoint(Path(args.checkpoint), device)
    if not isinstance(model, ResonanceTransformer):
        raise ValueError("bias_analysis requires a ResonanceTransformer checkpoint")

    model.eval()
    val_ds = StructuredSyntheticDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=min(args.val_samples, 100),
        num_modes=args.num_modes,
        seed=42 + 10_000,
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    all_qk_r_corr = []
    all_high_r_low_qk = []
    n_samples = 0
    max_samples = 50

    with torch.no_grad():
        for batch in loader:
            if n_samples >= max_samples:
                break
            input_ids = batch["input_ids"].to(device)
            x, resonance = model.embedding(input_ids)
            mask = model.causal_mask[:input_ids.size(1), :input_ids.size(1)].unsqueeze(0)

            # Compute QK^T for each layer
            for layer_idx, block in enumerate(model.blocks):
                ln_x = block.ln1(x)
                # Manual QKV
                qkv = block.attn.qkv(ln_x).reshape(
                    1, input_ids.size(1), 3, block.attn.n_heads, block.attn.head_dim
                ).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                qk = torch.matmul(q, k.transpose(-2, -1)) * block.attn.scale  # [1, H, S, S]

                # Resonance bias
                r = resonance.unsqueeze(1)  # [1, 1, S, S]
                w = block.attn.resonance_weight.view(1, -1, 1, 1)  # [1, H, 1, 1]

                # Correlation per head
                for h in range(block.attn.n_heads):
                    qk_h = qk[0, h].cpu().numpy().flatten()
                    r_h = r[0, 0].cpu().numpy().flatten()
                    # Mask out causal positions
                    causal_mask = mask[0].cpu().numpy().flatten().astype(bool)
                    qk_h = qk_h[causal_mask]
                    r_h = r_h[causal_mask]

                    if len(qk_h) > 1:
                        corr = np.corrcoef(np.abs(qk_h), np.abs(r_h))[0, 1]
                        if math.isfinite(corr):
                            all_qk_r_corr.append(float(corr))

                    # Find high-R, low-QK pairs
                    qk_norm = (qk_h - qk_h.mean()) / (qk_h.std() + 1e-12)
                    r_norm = (r_h - r_h.mean()) / (r_h.std() + 1e-12)
                    high_r_low_qk = (r_norm > 1.5) & (qk_norm < -0.5)
                    n_high_r = int(high_r_low_qk.sum())
                    if n_high_r > 0:
                        all_high_r_low_qk.append({
                            "layer": layer_idx,
                            "head": h,
                            "count": n_high_r,
                            "fraction": n_high_r / len(qk_h),
                        })

            n_samples += 1

    corr_mean = statistics.fmean(all_qk_r_corr) if all_qk_r_corr else 0.0
    corr_std = statistics.stdev(all_qk_r_corr) if len(all_qk_r_corr) > 1 else 0.0

    results = {
        "n_samples": n_samples,
        "qk_r_correlation_mean": corr_mean,
        "qk_r_correlation_std": corr_std,
        "n_heads_analyzed": len(all_qk_r_corr),
        "high_r_low_qk_instances": len(all_high_r_low_qk),
        "high_r_low_qk_details": all_high_r_low_qk[:20],
        "orthogonal_information": abs(corr_mean) < 0.5,
    }

    print("\n" + "=" * 60)
    print("Attention Bias Function Analysis")
    print("=" * 60)
    print(f"Mean |QK| vs |R| correlation: {corr_mean:.4f} ± {corr_std:.4f}")
    print(f"High-R/low-QK instances: {len(all_high_r_low_qk)}")
    print(f"Orthogonal information: {results['orthogonal_information']}")
    print("=" * 60)

    return results


# =============================================================================
# Experiment 6: Iso-Parameter Architecture Search
# =============================================================================

def run_iso_param_search(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Grid search over semantic/phase split at fixed parameter budget."""
    device = get_device(args.device)

    # Fixed budget
    target_params = args.target_params
    print(f"\nIso-parameter search: targeting ~{target_params:,} parameters")

    # Grid
    embed_dims = [128, 192, 256]
    frequencies = [8, 16, 32, 64]
    layers = [2, 4]

    train_ds = StructuredSyntheticDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.train_samples,
        num_modes=args.num_modes,
        seed=42,
    )
    val_ds = StructuredSyntheticDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=args.val_samples,
        num_modes=args.num_modes,
        seed=42 + 10_000,
    )

    results = []
    best_ppl = float("inf")
    best_config = None

    for embed_dim in embed_dims:
        for n_freq in frequencies:
            for n_layers in layers:
                # Adjust ff_dim to hit target
                ff_dim = args.ff_dim
                # Quick param estimate
                from foundational.iso_param import _count_params_at_dim
                est = _count_params_at_dim(
                    args.vocab_size, args.seq_len, embed_dim, n_layers,
                    args.n_heads, ff_dim, is_resonance=True, n_frequencies=n_freq,
                )
                if abs(est - target_params) > target_params * 0.3:
                    continue  # Skip if too far from target

                set_seed(42)
                config = ResonanceConfig(
                    name=f"iso_e{embed_dim}_f{n_freq}_l{n_layers}",
                    vocab_size=args.vocab_size,
                    max_seq_len=args.seq_len,
                    embed_dim=embed_dim,
                    n_layers=n_layers,
                    n_heads=args.n_heads,
                    ff_dim=ff_dim,
                    n_frequencies=n_freq,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )
                model = ResonanceTransformer(config).to(device)
                actual_params = count_params(model)

                print(f"\n  Config: embed={embed_dim}, freq={n_freq}, layers={n_layers}")
                print(f"  Params: {actual_params:,} (target: {target_params:,})")

                history = train_model(
                    model, config, train_ds, val_ds,
                    n_epochs=args.epochs,
                    device=device,
                    weight_decay=args.weight_decay,
                    log_interval=max(1, len(train_ds) // args.batch_size + 1),
                )

                final = evaluate_lm(model, val_ds, config, device)
                print(f"  Final PPL: {final['ppl']:.2f}")

                results.append({
                    "embed_dim": embed_dim,
                    "n_frequencies": n_freq,
                    "n_layers": n_layers,
                    "params": actual_params,
                    "ppl": final["ppl"],
                    "acc1": final["acc1"],
                })

                if final["ppl"] < best_ppl:
                    best_ppl = final["ppl"]
                    best_config = results[-1]

    # Baseline: pure semantic at same param count
    from foundational.iso_param import find_iso_param_dim
    std_dim = find_iso_param_dim(
        target_params, args.vocab_size, args.seq_len, args.n_layers,
        args.n_heads, args.ff_dim, is_resonance=False,
    )
    set_seed(42)
    std_config = StandardConfig(
        name="standard_iso",
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        embed_dim=std_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    std_model = StandardTransformer(std_config).to(device)
    std_history = train_model(
        std_model, std_config, train_ds, val_ds,
        n_epochs=args.epochs,
        device=device,
        weight_decay=args.weight_decay,
        log_interval=max(1, len(train_ds) // args.batch_size + 1),
    )
    std_final = evaluate_lm(std_model, val_ds, std_config, device)

    results.append({
        "embed_dim": std_dim,
        "n_frequencies": 0,
        "n_layers": args.n_layers,
        "params": count_params(std_model),
        "ppl": std_final["ppl"],
        "acc1": std_final["acc1"],
        "is_standard": True,
    })

    print("\n" + "=" * 60)
    print("Iso-Parameter Search Results")
    print("=" * 60)
    print(f"{'Embed':>6} {'Freq':>6} {'Layers':>6} {'Params':>10} {'PPL':>8}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["ppl"]):
        std_mark = " (std)" if r.get("is_standard") else ""
        print(f"{r['embed_dim']:>6} {r['n_frequencies']:>6} {r['n_layers']:>6} "
              f"{r['params']:>10,} {r['ppl']:>8.2f}{std_mark}")
    if best_config:
        print(f"\nBest config: embed={best_config['embed_dim']}, freq={best_config['n_frequencies']}, "
              f"layers={best_config['n_layers']}, PPL={best_config['ppl']:.2f}")
    print("=" * 60)

    return {
        "results": results,
        "best_config": best_config,
        "standard_baseline": {
            "embed_dim": std_dim,
            "params": count_params(std_model),
            "ppl": std_final["ppl"],
        },
    }


# =============================================================================
# Experiment 7: Beta-Reduction Equivalence Retrieval
# =============================================================================

def run_beta_reduction(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Contrastive learning on beta-reduction (term, normal_form) pairs.

    The core realizability-topos test: can the model learn that a term
    and its beta-normal form are semantically equivalent?
    """
    from synthetic.beta_reduction_dataset import BetaReductionDataset
    from synthetic.contrastive_trainer import ProgramTokenizer
    from synthetic.vectorized_contrastive import VectorizedContrastiveLoss

    device = get_device(args.device)
    set_seed(args.proof_seed)

    print("\nGenerating beta-reduction datasets...")
    train_ds = BetaReductionDataset(
        n_pairs=args.beta_train_pairs,
        max_depth=args.max_term_depth,
        max_size=args.max_term_size,
        min_trace_length=2,
        generator_seed=args.proof_seed,
    )
    val_ds = BetaReductionDataset(
        n_pairs=args.beta_val_pairs,
        max_depth=args.max_term_depth,
        max_size=args.max_term_size,
        min_trace_length=2,
        generator_seed=args.proof_seed + 1,
    )

    stats = train_ds.statistics()
    print(f"  Train stats: {stats}")

    tokenizer = ProgramTokenizer(max_length=args.program_max_length)

    # Build encoders: wrapper around Standard/Resonance transformers
    class TermEncoder(nn.Module):
        def __init__(self, backbone: nn.Module, embed_dim: int) -> None:
            super().__init__()
            self.backbone = backbone
            self.projection = nn.Linear(embed_dim, args.proof_d_model)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            # tokens: [B, S]
            pad_mask = tokens != tokenizer.pad_id  # [B, S]

            if isinstance(self.backbone, ResonanceTransformer):
                x, resonance = self.backbone.embedding(tokens)
                mask = self.backbone.causal_mask[:tokens.size(1), :tokens.size(1)].unsqueeze(0)
                for block in self.backbone.blocks:
                    x = block(x, resonance, mask)
                x = self.backbone.ln_final(x)
            else:
                x = self.backbone.token_embed(tokens) + self.backbone.pos_embed(
                    torch.arange(tokens.size(1), device=tokens.device)
                )
                mask = self.backbone.causal_mask[:tokens.size(1), :tokens.size(1)].unsqueeze(0)
                for block in self.backbone.blocks:
                    x = block(x, mask)
                x = self.backbone.ln_final(x)

            # Mean pooling over non-pad positions
            x = x * pad_mask.unsqueeze(-1).float()
            lengths = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = x.sum(dim=1) / lengths  # [B, embed_dim]
            return self.projection(pooled)  # [B, proof_d_model]

    conditions = {
        "standard": lambda: TermEncoder(
            StandardTransformer(StandardConfig(
                vocab_size=tokenizer.vocab_size,
                max_seq_len=args.program_max_length,
                embed_dim=args.proof_d_model,
                n_layers=args.proof_layers,
                n_heads=args.proof_heads,
                ff_dim=args.proof_d_model * 4,
                dropout=args.dropout,
                batch_size=args.proof_batch_size,
            )),
            args.proof_d_model,
        ),
        "resonance": lambda: TermEncoder(
            ResonanceTransformer(ResonanceConfig(
                vocab_size=tokenizer.vocab_size,
                max_seq_len=args.program_max_length,
                embed_dim=args.proof_d_model,
                n_layers=args.proof_layers,
                n_heads=args.proof_heads,
                ff_dim=args.proof_d_model * 4,
                n_frequencies=args.n_frequencies,
                dropout=args.dropout,
                batch_size=args.proof_batch_size,
                use_phase_stream=True,
                use_resonance_bias=True,
            )),
            args.proof_d_model,
        ),
    }

    all_results = {}

    for name, model_fn in conditions.items():
        print(f"\n{'='*60}")
        print(f"Beta-Reduction: {name}")
        print(f"{'='*60}")

        set_seed(args.proof_seed)
        model = model_fn().to(device)
        print(f"  Params: {count_params(model):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.proof_learning_rate)
        contrastive = VectorizedContrastiveLoss(temperature=args.temperature)

        def encode_batch(texts: list[str]) -> torch.Tensor:
            tokens = tokenizer.batch_encode(texts, device=str(device))
            return model(tokens)

        # Training
        history = []
        for epoch in range(args.proof_epochs):
            model.train()
            rng = random.Random(args.proof_seed + epoch)
            indices = list(range(len(train_ds)))
            rng.shuffle(indices)

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(indices), args.proof_batch_size):
                batch_idx = indices[i:i + args.proof_batch_size]
                originals = [train_ds.pairs[j]["original"] for j in batch_idx]
                nfs = [train_ds.pairs[j]["normal_form"] for j in batch_idx]

                orig_emb = encode_batch(originals)  # [B, D]
                nf_emb = encode_batch(nfs)  # [B, D]

                # Cross-similarity: originals vs normal forms
                # Positive: diagonal pairs (orig_i, nf_i)
                # Negative: off-diagonal pairs
                B = orig_emb.size(0)
                all_emb = torch.cat([orig_emb, nf_emb], dim=0)  # [2B, D]
                positive_mask = torch.zeros(2 * B, 2 * B, dtype=torch.bool, device=device)
                for j in range(B):
                    positive_mask[j, B + j] = True
                    positive_mask[B + j, j] = True

                loss = contrastive(all_emb, positive_mask)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append(avg_loss)
            print(f"  Epoch {epoch + 1}/{args.proof_epochs}: loss={avg_loss:.4f}")

        # Evaluation: retrieval
        model.eval()
        with torch.no_grad():
            all_origs = [p["original"] for p in val_ds.pairs]
            all_nfs = [p["normal_form"] for p in val_ds.pairs]
            orig_emb = encode_batch(all_origs)  # [N, D]
            nf_emb = encode_batch(all_nfs)  # [N, D]

            # Normalize
            orig_emb = F.normalize(orig_emb, dim=-1)
            nf_emb = F.normalize(nf_emb, dim=-1)

            # Similarity: each original vs all NFs
            sim = orig_emb @ nf_emb.T  # [N, N]

            results = {}
            for k in [1, 3, 5]:
                _, topk = sim.topk(k, dim=-1)
                correct = 0
                for i in range(len(val_ds.pairs)):
                    if i in topk[i].cpu().tolist():
                        correct += 1
                results[f"recall@{k}"] = correct / len(val_ds.pairs)

            # Mean reciprocal rank
            ranks = []
            for i in range(len(val_ds.pairs)):
                # Find rank of correct NF
                order = sim[i].argsort(descending=True)
                rank = (order == i).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(1.0 / rank)
            results["mrr"] = sum(ranks) / len(ranks)

            # Positive-negative gap
            pos_sims = sim.diagonal().cpu().tolist()
            neg_mask = ~torch.eye(len(val_ds.pairs), dtype=torch.bool, device=device)
            neg_sims = sim[neg_mask].cpu().tolist()
            results["pos_mean"] = sum(pos_sims) / len(pos_sims)
            results["neg_mean"] = sum(neg_sims) / len(neg_sims)
            results["gap"] = results["pos_mean"] - results["neg_mean"]

        print(f"  Recall@1: {results['recall@1']:.4f}")
        print(f"  Recall@5: {results['recall@5']:.4f}")
        print(f"  MRR: {results['mrr']:.4f}")
        print(f"  Gap: {results['gap']:.4f}")

        all_results[name] = results

    return all_results


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    output_dir: Path,
    results: dict[str, Any],
) -> Path:
    """Generate a unified Markdown report from all experiment results."""
    lines = [
        "# Foundational Experiment Suite Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
    ]

    # Experiment 1: Ablation
    if "ablation" in results:
        ab = results["ablation"]
        lines.extend([
            "### 1. Multi-Seed Ablation (Iso-Parameter)",
            "",
            "| Condition | Params | PPL Mean | PPL Std | Ratio vs Std |",
            "|---|---:|---:|---:|---:|",
        ])
        for name, row in sorted(ab.get("summary", {}).items()):
            if name == "effect_size":
                continue
            lines.append(
                f"| {name} | {row.get('params', 0):,} | "
                f"{row.get('ppl_mean', 0):.2f} | {row.get('ppl_std', 0):.2f} | "
                f"{row.get('ppl_ratio_vs_standard', 1.0):.3f} |"
            )
        if "effect_size" in ab.get("summary", {}):
            es = ab["summary"]["effect_size"]
            lines.append(f"\n**Effect size (Cohen's d):** {es.get('cohens_d', 0):.3f}")
        lines.append("")

    # Experiment 2: Geometry
    if "geometry" in results:
        geo = results["geometry"]
        lines.extend([
            "### 2. Phase Geometry & Compressibility",
            "",
            f"- Semantic effective rank (95%): {geo.get('semantic_geometry', {}).get('effective_rank_95', 'N/A')}",
            f"- Phase effective rank (95%): {geo.get('phase_geometry', {}).get('effective_rank_95', 'N/A')}",
            f"- R² (phase → semantic): {geo.get('cross_predictability', {}).get('r2_phase_to_semantic', 0):.4f}",
            f"- R² (random → semantic): {geo.get('cross_predictability', {}).get('r2_random_to_semantic', 0):.4f}",
            f"- Blend mean: {geo.get('blend_stats', {}).get('mean', 0):.3f}",
            "",
        ])

    # Experiment 3: Perturbation
    if "perturbation" in results:
        pert = results["perturbation"]
        lines.extend([
            "### 3. Perturbation Intervention Battery",
            "",
        ])
        for condition, cond_results in pert.items():
            if condition == "phase_specificity":
                continue
            lines.append(f"**{condition}:**")
            for ptype, vals in cond_results.items():
                if ptype == "clean":
                    lines.append(f"- Clean PPL: {vals.get('ppl', 0):.2f}")
                elif isinstance(vals, dict) and "delta_ppl" in vals:
                    lines.append(f"- {ptype}: PPL={vals['ppl']:.2f}, Δ={vals['delta_ppl']:+.2f}")
            lines.append("")
        if "phase_specificity" in pert:
            lines.append("**Phase specificity ratios:**\n")
            for ptype, vals in pert["phase_specificity"].items():
                lines.append(f"- {ptype}: {vals.get('specificity_ratio', 0):.2f}x")
            lines.append("")

    # Experiment 4: Retrieval
    if "retrieval" in results:
        ret = results["retrieval"]
        lines.extend([
            "### 4. Equivalence Retrieval Benchmark",
            "",
            "| Condition | Sem R@1 | Sem R@5 | Phase R@1 | Phase R@5 | Sem Gap | Phase Gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, row in sorted(ret.items()):
            lines.append(
                f"| {name} | {row.get('semantic_recall@1', 0):.4f} | "
                f"{row.get('semantic_recall@5', 0):.4f} | "
                f"{row.get('phase_recall@1', 0):.4f} | {row.get('phase_recall@5', 0):.4f} | "
                f"{row.get('semantic_gap', 0):.4f} | {row.get('phase_gap', 0):.4f} |"
            )
        lines.append("")

    # Experiment 5: Bias analysis
    if "bias_analysis" in results:
        bias = results["bias_analysis"]
        lines.extend([
            "### 5. Attention Bias Function Analysis",
            "",
            f"- Mean |QK| vs |R| correlation: {bias.get('qk_r_correlation_mean', 0):.4f}",
            f"- Orthogonal information: {bias.get('orthogonal_information', False)}",
            f"- High-R/low-QK instances: {bias.get('high_r_low_qk_instances', 0)}",
            "",
        ])

    # Experiment 7: Beta-reduction equivalence retrieval
    if "beta_reduction" in results:
        beta = results["beta_reduction"]
        lines.extend([
            "### 7. Beta-Reduction Equivalence Retrieval",
            "",
            "| Condition | R@1 | R@3 | R@5 | MRR | Pos-Mean | Neg-Mean | Gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name, row in sorted(beta.items()):
            lines.append(
                f"| {name} | {row.get('recall@1', 0):.4f} | "
                f"{row.get('recall@3', 0):.4f} | {row.get('recall@5', 0):.4f} | "
                f"{row.get('mrr', 0):.4f} | {row.get('pos_mean', 0):.4f} | "
                f"{row.get('neg_mean', 0):.4f} | {row.get('gap', 0):.4f} |"
            )
        lines.append("")

    # Experiment 6: Iso-param search
    if "iso_param" in results:
        iso = results["iso_param"]
        lines.extend([
            "### 6. Iso-Parameter Architecture Search",
            "",
            f"- Best config: {iso.get('best_config', {})}",
            f"- Standard baseline PPL: {iso.get('standard_baseline', {}).get('ppl', 0):.2f}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*Generated by found_suite.py*",
    ])

    report_path = output_dir / "foundational_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=[
        "ablation", "geometry", "perturbation", "retrieval", "bias_analysis",
        "iso_param", "beta_reduction", "all"
    ], default="all")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/foundational"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for geometry/bias_analysis")
    parser.add_argument("--checkpoint_dir", default=None, help="Checkpoint dir for perturbation")

    # Data & training
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 42])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_samples", type=int, default=3000)
    parser.add_argument("--val_samples", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--num_modes", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    # Model architecture
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--phase_init_std", type=float, default=0.3)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)

    # Proof-walk / retrieval
    parser.add_argument("--proof_seed", type=int, default=123)
    parser.add_argument("--proof_train_groups", type=int, default=64)
    parser.add_argument("--proof_val_groups", type=int, default=32)
    parser.add_argument("--programs_per_group", type=int, default=6)
    parser.add_argument("--max_term_depth", type=int, default=4)
    parser.add_argument("--max_term_size", type=int, default=10)
    parser.add_argument("--program_max_length", type=int, default=96)
    parser.add_argument("--proof_d_model", type=int, default=64)
    parser.add_argument("--proof_layers", type=int, default=2)
    parser.add_argument("--proof_heads", type=int, default=2)
    parser.add_argument("--proof_batch_size", type=int, default=8)
    parser.add_argument("--proof_epochs", type=int, default=10)
    parser.add_argument("--proof_learning_rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)

    # Iso-param search
    parser.add_argument("--target_params", type=int, default=500_000)

    # Beta-reduction
    parser.add_argument("--beta_train_pairs", type=int, default=1000)
    parser.add_argument("--beta_val_pairs", type=int, default=200)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_deterministic(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(_jsonify(vars(args)), f, indent=2)

    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = ["ablation", "geometry", "perturbation", "retrieval", "bias_analysis", "iso_param", "beta_reduction"]
    else:
        experiments_to_run = [args.experiment]

    all_results = {}

    for exp_name in experiments_to_run:
        print(f"\n{'#'*70}")
        print(f"# Running experiment: {exp_name}")
        print(f"{'#'*70}")

        exp_output = args.output_dir / exp_name
        exp_output.mkdir(parents=True, exist_ok=True)

        try:
            if exp_name == "ablation":
                result = run_ablation(args, exp_output)
            elif exp_name == "geometry":
                result = run_geometry(args, exp_output)
            elif exp_name == "perturbation":
                result = run_perturbation(args, exp_output)
            elif exp_name == "retrieval":
                result = run_retrieval(args, exp_output)
            elif exp_name == "bias_analysis":
                result = run_bias_analysis(args, exp_output)
            elif exp_name == "iso_param":
                result = run_iso_param_search(args, exp_output)
            elif exp_name == "beta_reduction":
                result = run_beta_reduction(args, exp_output)
            else:
                raise ValueError(f"Unknown experiment: {exp_name}")

            all_results[exp_name] = result

            with open(exp_output / "results.json", "w") as f:
                json.dump(_jsonify(result), f, indent=2)

        except Exception as e:
            print(f"ERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {"error": str(e)}

    # Generate unified report
    report_path = generate_report(args.output_dir, all_results)
    print(f"\n{'='*70}")
    print(f"Report written to: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
