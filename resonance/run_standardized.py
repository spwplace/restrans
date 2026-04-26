#!/usr/bin/env python3
"""
Standardized Benchmark Run (GPT-2 BPE)
======================================

Runs Resonance and Standard transformers with GPT-2 BPE tokenization
for direct comparability with einygpt and the TinyStories paper.

Key differences from the default word-level experiment:
  - GPT-2 BPE tokenizer (vocab=50,257)
  - Iso-parameter matching: resonance dims adjusted to match baseline param count
  - Full TinyStories or a fixed sample count
  - All seeds identical, all hyperparameters identical

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python run_standardized.py \
        --condition baseline --epochs 20 --device cuda

    HSA_OVERRIDE_GFX_VERSION=11.0.0 python run_standardized.py \
        --condition resonance --epochs 20 --device cuda

Recommended iso-param scale (match einygpt ~6.9M params with GPT-2 vocab):
  - Standard:  embed_dim=384, n_layers=6, n_heads=6  → ~6.9M params
  - Resonance: embed_dim=384, n_layers=6, n_heads=6  → ~7.1M params (+2.5%)

To get exact iso-param, tune embed_dim slightly (e.g., 376d for resonance).
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def estimate_params(vocab_size: int, embed_dim: int, n_layers: int, n_heads: int, ff_dim: int, n_frequencies: int = 0) -> int:
    """Rough parameter count estimator."""
    embed = vocab_size * embed_dim
    phase = vocab_size * n_frequencies
    # QKV + out projection: 4 * D^2 per layer
    attn = n_layers * (4 * embed_dim * embed_dim)
    # FFN: 2 * D * ff_dim per layer
    ff = n_layers * (2 * embed_dim * ff_dim)
    # LN params are negligible
    return embed + phase + attn + ff + embed_dim  # lm_head tied, + pos_embed


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardized Benchmark Run")
    parser.add_argument("--condition", choices=["baseline", "resonance"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_samples", type=int, default=100000)
    parser.add_argument("--val_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256, help="GPT-2 uses longer contexts")
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--ff_dim", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="resonance/outputs/standardized")
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    vocab_size = 50257  # GPT-2 vocab
    std_params = estimate_params(vocab_size, args.embed_dim, args.n_layers, args.n_heads, args.ff_dim)
    res_params = estimate_params(vocab_size, args.embed_dim, args.n_layers, args.n_heads, args.ff_dim, args.n_frequencies)

    print(f"Estimated params — Standard: {std_params:,}  Resonance: {res_params:,}  (+{(res_params/std_params-1)*100:.1f}%)")

    cmd = [
        sys.executable,
        "resonance/experiment_text.py",
        "--condition", args.condition,
        "--epochs", str(args.epochs),
        "--train_samples", str(args.train_samples),
        "--val_samples", str(args.val_samples),
        "--batch_size", str(args.batch_size),
        "--seq_len", str(args.seq_len),
        "--embed_dim", str(args.embed_dim),
        "--n_layers", str(args.n_layers),
        "--n_heads", str(args.n_heads),
        "--ff_dim", str(args.ff_dim),
        "--seed", str(args.seed),
        "--device", args.device,
        "--tokenizer", "gpt2",
        "--output_dir", args.output_dir,
    ]

    if args.condition == "resonance":
        cmd.extend(["--use_phase_stream", "true", "--use_resonance_bias", "true"])

    print(f"\nCommand:\n{' '.join(cmd)}\n")

    if args.dry_run:
        print("Dry run — not executing.")
        return

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
