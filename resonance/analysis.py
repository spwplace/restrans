#!/usr/bin/env python3
"""
Resonance Transformer Checkpoint Analysis
=========================================

Loads a trained checkpoint and computes:
  1. Parameter-level statistics (sparsity, norms)
  2. Embedding geometry (PCA effective rank, variance distribution)
  3. Resonance matrix properties (sparsity, rank, diagonal patterns)
  4. Cross-predictability (linear probe from phase → semantic)
  5. Attention pattern inspection (resonance bias vs QK^T contribution)

Usage:
    python analysis.py --checkpoint resonance/outputs/experiment_resonance/checkpoints/latest.pt
    python analysis.py --checkpoint latest.pt --output_dir analysis_out/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def load_checkpoint(path: Path, device: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint and return its contents."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def param_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute basic parameter statistics."""
    stats = {}
    total = 0
    nonzero = 0
    for name, param in state_dict.items():
        total += param.numel()
        nonzero += (param != 0).sum().item()
        stats[f"{name}_mean"] = float(param.mean())
        stats[f"{name}_std"] = float(param.std())
        stats[f"{name}_norm"] = float(param.norm())
    stats["total_params"] = total
    stats["nonzero_params"] = nonzero
    stats["sparsity"] = 1.0 - (nonzero / max(total, 1))
    return stats


def embedding_geometry(state_dict: dict[str, torch.Tensor], vocab_size: int) -> dict[str, Any]:
    """Analyze semantic and phase embedding geometry via PCA."""
    results: dict[str, Any] = {}

    # Try to find semantic and phase embeddings
    sem_key = None
    phase_key = None
    for k in state_dict:
        if "semantic" in k and "weight" in k:
            sem_key = k
        if "phase" in k and "proj" not in k and "weight" in k:
            phase_key = k

    if sem_key is None:
        return {"error": "No semantic embedding found"}

    sem = state_dict[sem_key]  # [V, D]
    results["semantic_shape"] = list(sem.shape)

    # PCA on semantic embeddings
    sem_centered = sem - sem.mean(dim=0)
    u, s, vh = torch.svd(sem_centered)
    sem_var = (s**2) / (s**2).sum()
    results["semantic_pca_var_top5"] = sem_var[:5].tolist()
    results["semantic_pca_rank_95"] = int((sem_var.cumsum(dim=0) < 0.95).sum().item() + 1)
    results["semantic_pca_rank_99"] = int((sem_var.cumsum(dim=0) < 0.99).sum().item() + 1)

    if phase_key is not None:
        phase = state_dict[phase_key]  # [V, F]
        results["phase_shape"] = list(phase.shape)
        phase_centered = phase - phase.mean(dim=0)
        u_p, s_p, vh_p = torch.svd(phase_centered)
        phase_var = (s_p**2) / (s_p**2).sum()
        results["phase_pca_var_top5"] = phase_var[:5].tolist()
        results["phase_pca_rank_95"] = int((phase_var.cumsum(dim=0) < 0.95).sum().item() + 1)
        results["phase_pca_rank_99"] = int((phase_var.cumsum(dim=0) < 0.99).sum().item() + 1)

        # Cross-predictability: linear ridge regression from phase → semantic
        X = phase.numpy()
        Y = sem.numpy()
        # Simple least-squares solution
        XtX = X.T @ X
        XtY = X.T @ Y
        # Add small ridge
        ridge = 1e-4 * np.eye(X.shape[1])
        W = np.linalg.solve(XtX + ridge, XtY)
        Y_pred = X @ W
        ss_res = ((Y - Y_pred) ** 2).sum()
        ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot
        results["phase_to_semantic_r2"] = float(r2)

    return results


def resonance_matrix_stats(model: torch.nn.Module, vocab_size: int, seq_len: int = 128) -> dict[str, Any]:
    """Compute statistics about the resonance matrix."""
    results: dict[str, Any] = {}

    # Try to compute resonance matrix from a dummy forward pass
    try:
        device = next(model.parameters()).device
        dummy = torch.arange(min(vocab_size, 1000), device=device).unsqueeze(0)  # [1, V']

        # Hook into embedding layer to capture resonance
        resonance_hook = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                resonance_hook["matrix"] = output[1].detach().cpu()

        # Find ResonanceEmbedding
        for name, module in model.named_modules():
            if "ResonanceEmbedding" in module.__class__.__name__:
                handle = module.register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(dummy)
                handle.remove()
                break

        if "matrix" in resonance_hook:
            R = resonance_hook["matrix"][0]  # [S, S]
            results["resonance_mean"] = float(R.mean())
            results["resonance_std"] = float(R.std())
            results["resonance_min"] = float(R.min())
            results["resonance_max"] = float(R.max())
            results["resonance_diag_mean"] = float(R.diag().mean())
            # Fraction of off-diagonal entries with |R| > 0.1
            off_diag = R.clone()
            off_diag.fill_diagonal_(0)
            results["resonance_strong_connections"] = float((off_diag.abs() > 0.1).sum().item() / max(off_diag.numel(), 1))
            # Effective rank
            u, s, vh = torch.svd(R)
            s_norm = s / s.sum()
            results["resonance_effective_rank"] = float((s_norm**2).sum().pow(-1).item())
        else:
            results["note"] = "No resonance matrix captured (model may not be ResonanceTransformer)"
    except Exception as e:
        results["error"] = str(e)

    return results


def blend_and_weight_stats(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Report learned blend and resonance weight values."""
    results = {}
    for name, param in state_dict.items():
        if "blend" in name or "alpha" in name:
            results["blend_mean"] = float(param.mean())
            results["blend_std"] = float(param.std())
            if param.numel() < 50:
                results["blend_values"] = param.tolist()
        if "resonance_weight" in name:
            results["resonance_weight_mean"] = float(param.mean())
            results["resonance_weight_std"] = float(param.std())
            if param.numel() < 50:
                results["resonance_weight_values"] = param.tolist()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Resonance Transformer checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to write JSON results")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size for resonance analysis")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length for resonance analysis")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)

    state_dict = ckpt.get("model_state", ckpt)
    results: dict[str, Any] = {}

    # 1. Parameter stats
    print("Computing parameter statistics...")
    results["params"] = param_stats(state_dict)

    # 2. Embedding geometry
    print("Computing embedding geometry...")
    results["geometry"] = embedding_geometry(state_dict, args.vocab_size)

    # 3. Blend and weight stats
    print("Computing blend / weight statistics...")
    results["controls"] = blend_and_weight_stats(state_dict)

    # 4. Resonance matrix stats (need to load model)
    print("Computing resonance matrix statistics...")
    # Try to reconstruct model from checkpoint
    try:
        from resonance.models import ResonanceTransformer, StandardTransformer
        from resonance.config import ResonanceConfig, StandardConfig

        config = ckpt.get("config")
        if config is None:
            results["resonance"] = {"note": "No config in checkpoint; cannot reconstruct model"}
        elif isinstance(config, ResonanceConfig):
            model = ResonanceTransformer(config)
            model.load_state_dict(state_dict)
            model.eval()
            results["resonance"] = resonance_matrix_stats(model, args.vocab_size, args.seq_len)
        else:
            results["resonance"] = {"note": "StandardTransformer has no resonance matrix"}
    except Exception as e:
        results["resonance"] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    # Save
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{args.checkpoint.stem}_analysis.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
