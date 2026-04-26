#!/usr/bin/env python3
"""Quick mechanical comparison of all resonance kernels.

Trains a tiny model for 2 epochs with each kernel on structured synthetic
LM data and reports trainability + final loss.  This is a fast smoke test
(~2 minutes total) to spot kernels that are numerically unstable or
mechanically useless.

Usage:
    uv run python kernel_quick_compare.py --device mps
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from resonance.config import ResonanceConfig
from resonance.device import get_device, set_seed
from resonance.kernels import list_kernels
from resonance.models import ResonanceTransformer
from resonance.training import train_model
from train import StructuredSyntheticDataset


def evaluate_kernel(
    kernel_name: str,
    device: torch.device,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    config = ResonanceConfig(
        name=f"kernel_{kernel_name}",
        vocab_size=256,
        max_seq_len=64,
        embed_dim=128,
        n_layers=2,
        n_heads=4,
        ff_dim=256,
        n_frequencies=16,
        resonance_kernel=kernel_name,
        dropout=0.0,
        batch_size=32,
        learning_rate=3e-4,
    )
    model = ResonanceTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_ds = StructuredSyntheticDataset(
        vocab_size=256, seq_len=64, num_samples=500, num_modes=8, seed=seed,
    )
    val_ds = StructuredSyntheticDataset(
        vocab_size=256, seq_len=64, num_samples=100, num_modes=8, seed=seed + 10_000,
    )

    start = time.perf_counter()
    history = train_model(
        model, config, train_ds, val_ds,
        n_epochs=2, device=device, log_interval=99999,
    )
    elapsed = time.perf_counter() - start

    # Mechanical diagnostics on random phases
    with torch.no_grad():
        sample_tokens = torch.randint(0, 256, (4, 16), device=device)
        _, r = model.embedding(sample_tokens)
        r_mean = float(r.mean().item())
        r_std = float(r.std(unbiased=False).item())
        # Effective rank of first batch item (compute on CPU for MPS compatibility)
        r_cpu = r[0].float().cpu()
        s = torch.linalg.svdvals(r_cpu)
        s = s[s > 1e-8]
        probs = s / s.sum()
        eff_rank = float(torch.exp(-(probs * torch.log(probs + 1e-12)).sum()).item())

    return {
        "kernel": kernel_name,
        "params": n_params,
        "train_loss_1": history["train_loss"][0],
        "train_loss_2": history["train_loss"][1],
        "val_ppl_1": history["val_ppl"][0],
        "val_ppl_2": history["val_ppl"][1],
        "r_mean": r_mean,
        "r_std": r_std,
        "eff_rank": eff_rank,
        "sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/kernel_compare"))
    args = parser.parse_args()

    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for kernel in list_kernels():
        print(f"\n{'='*60}")
        print(f"Kernel: {kernel}")
        print(f"{'='*60}")
        try:
            row = evaluate_kernel(kernel, device)
            results.append(row)
            print(
                f"  PPL: {row['val_ppl_1']:.1f} → {row['val_ppl_2']:.1f}  |  "
                f"R: μ={row['r_mean']:.3f} σ={row['r_std']:.3f}  |  "
                f"rank={row['eff_rank']:.2f}  |  {row['sec']:.1f}s"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"kernel": kernel, "error": str(e)})

    # Sort by final PPL
    good = [r for r in results if "val_ppl_2" in r]
    good.sort(key=lambda x: x["val_ppl_2"])

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by final val PPL)")
    print("=" * 70)
    print(f"{'Kernel':<20} {'PPL₁':>8} {'PPL₂':>8} {'R_μ':>7} {'R_σ':>7} {'Rank':>6} {'Sec':>5}")
    print("-" * 70)
    for r in good:
        print(
            f"{r['kernel']:<20} {r['val_ppl_1']:>8.1f} {r['val_ppl_2']:>8.1f} "
            f"{r['r_mean']:>7.3f} {r['r_std']:>7.3f} {r['eff_rank']:>6.2f} {r['sec']:>5.1f}"
        )
    print("=" * 70)

    # Save
    import json
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
