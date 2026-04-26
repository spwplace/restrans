#!/usr/bin/env python3
"""
Resonance Ablation Grid
=======================

Runs a systematic ablation of the ResonanceTransformer to isolate which
components drive the performance difference vs StandardTransformer.

Conditions (all with identical seed, data, and hyperparameters):
  1. baseline       — StandardTransformer (no extra params)
  2. resonance_full — ResonanceTransformer (phase stream + resonance bias)
  3. no_bias        — ResonanceTransformer with resonance_bias=False
  4. no_phase       — ResonanceTransformer with use_phase_stream=False
  5. neither        — ResonanceTransformer with both disabled
                       (tests if extra parameters alone help)

Usage:
    python run_ablations.py --seed 42 --epochs 10 --device cuda
    python run_ablations.py --seed 42 --epochs 10 --device mps --train_samples 10000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ABLATIONS = [
    ("baseline", {}),
    ("resonance_full", {"use_phase_stream": "true", "use_resonance_bias": "true"}),
    ("no_bias", {"use_phase_stream": "true", "use_resonance_bias": "false"}),
    ("no_phase", {"use_phase_stream": "false", "use_resonance_bias": "true"}),
    ("neither", {"use_phase_stream": "false", "use_resonance_bias": "false"}),
]


def run_condition(
    name: str,
    extra_args: dict[str, str],
    base_args: argparse.Namespace,
) -> None:
    """Launch a single ablation condition."""
    out_dir = Path(base_args.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "resonance/experiment_text.py",
        "--condition", "resonance" if extra_args else "baseline",
        "--epochs", str(base_args.epochs),
        "--train_samples", str(base_args.train_samples),
        "--val_samples", str(base_args.val_samples),
        "--batch_size", str(base_args.batch_size),
        "--seq_len", str(base_args.seq_len),
        "--embed_dim", str(base_args.embed_dim),
        "--n_layers", str(base_args.n_layers),
        "--n_heads", str(base_args.n_heads),
        "--seed", str(base_args.seed),
        "--device", base_args.device,
        "--output_dir", str(out_dir),
    ]

    for k, v in extra_args.items():
        cmd.extend([f"--{k}", v])

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Output: {out_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Resonance Ablation Grid")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_samples", type=int, default=100000)
    parser.add_argument("--val_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="resonance/outputs/ablations")
    parser.add_argument("--conditions", default="all", help="Comma-separated list or 'all'")
    args = parser.parse_args()

    conditions = ABLATIONS if args.conditions == "all" else [
        (name, flags) for name, flags in ABLATIONS if name in args.conditions.split(",")
    ]

    print(f"Ablation grid: {len(conditions)} conditions")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Output root: {args.output_dir}")

    for name, flags in conditions:
        run_condition(name, flags, args)

    print("\n" + "=" * 60)
    print("All ablations complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
