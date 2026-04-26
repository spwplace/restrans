#!/usr/bin/env python3
"""Probe resonance-kernel scale before training.

The additive attention-bias path can be a no-op if phase vectors are too close
together or if the resonance weight is too small relative to QK logits.  This
script instantiates small random resonance models and reports the sampled
resonance statistics plus the attention-probability delta caused by the bias.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from resonance.config import ResonanceConfig
from resonance.device import set_seed
from resonance.interpretability import resonance_diagnostics, trace_resonance_forward
from resonance.models import ResonanceTransformer


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonify(v) for v in value]
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else str(value)
    return value


def run_one(
    *,
    seed: int,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    embed_dim: int,
    n_heads: int,
    n_frequencies: int,
    phase_init_std: float,
    resonance_attn_weight: float,
    center_resonance: bool,
    normalize_resonance: bool,
) -> dict[str, float]:
    set_seed(seed)
    config = ResonanceConfig(
        name="kernel_sweep",
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        embed_dim=embed_dim,
        n_layers=1,
        n_heads=n_heads,
        ff_dim=4 * embed_dim,
        n_frequencies=n_frequencies,
        phase_init_std=phase_init_std,
        resonance_attn_weight=resonance_attn_weight,
        resonance_blend=0.3,
        dropout=0.0,
        use_phase_stream=True,
        use_resonance_bias=True,
        center_resonance=center_resonance,
        normalize_resonance=normalize_resonance,
    )
    model = ResonanceTransformer(config)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    trace = trace_resonance_forward(model, tokens)
    metrics = resonance_diagnostics(trace)
    metrics.update(
        {
            "seed": float(seed),
            "n_frequencies": float(n_frequencies),
            "phase_init_std": phase_init_std,
            "resonance_attn_weight": resonance_attn_weight,
            "center_resonance": float(center_resonance),
            "normalize_resonance": float(normalize_resonance),
        }
    )
    return metrics


def write_report(path: Path, rows: list[dict[str, float]]) -> None:
    lines = [
        "# Resonance Kernel Sweep",
        "",
        "| Centered | Normalized | F | Phase Std | Weight | R Mean | R Std | Rank | Attn Delta | Logit Abs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['center_resonance'])} | {int(row['normalize_resonance'])} | "
            f"{int(row['n_frequencies'])} | {row['phase_init_std']:.3f} | "
            f"{row['resonance_attn_weight']:.3f} | {row['resonance_mean']:.4f} | "
            f"{row['resonance_std']:.4f} | {row['resonance_effective_rank_first']:.4f} | "
            f"{row['attention_delta_abs_mean']:.6f} | {row['resonance_logit_abs_mean']:.4f} |"
        )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/kernel_sweep"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--phase_stds", type=float, nargs="+", default=[0.05, 0.1, 0.3, 0.75, 1.2, 2.0])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--frequencies", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--centered", action="store_true")
    parser.add_argument("--normalized", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for n_frequencies in args.frequencies:
            for phase_std in args.phase_stds:
                for weight in args.weights:
                    rows.append(
                        run_one(
                            seed=seed,
                            vocab_size=args.vocab_size,
                            seq_len=args.seq_len,
                            batch_size=args.batch_size,
                            embed_dim=args.embed_dim,
                            n_heads=args.n_heads,
                            n_frequencies=n_frequencies,
                            phase_init_std=phase_std,
                            resonance_attn_weight=weight,
                            center_resonance=args.centered or args.normalized,
                            normalize_resonance=args.normalized,
                        )
                    )
    rows.sort(
        key=lambda row: (
            int(row["center_resonance"]),
            int(row["normalize_resonance"]),
            int(row["n_frequencies"]),
            row["phase_init_std"],
            row["resonance_attn_weight"],
        )
    )
    fieldnames = list(rows[0].keys()) if rows else []
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "results.json").write_text(json.dumps(jsonify(rows), indent=2))
    write_report(args.output_dir / "report.md", rows)
    print(f"Wrote {args.output_dir / 'results.csv'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
