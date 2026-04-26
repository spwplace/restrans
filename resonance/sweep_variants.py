#!/usr/bin/env python3
"""Focused sweep of resonance architecture variants on beta-reduction data.

Tests combinations of phase embeddings, kernels, bias modes, and init presets
to find which configurations actually show signal on topology-bearing data.

Grid (48 combos):
    Phase:     real, fourier_fixed, complex_angle
    Kernel:    cosine, complex_magnitude, rbf, bilinear
    Bias:      additive, residual_gate
    Preset:    default, strong

Usage:
    uv run python sweep_variants.py --device mps --output_dir outputs/sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from resonance.config import ResonanceConfig
from resonance.device import get_device, set_seed
from resonance.models import ResonanceTransformer
from synthetic.beta_reduction_dataset import BetaReductionDataset
from synthetic.contrastive_trainer import ProgramTokenizer
from synthetic.vectorized_contrastive import VectorizedContrastiveLoss


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_condition(
    phase: str,
    kernel: str,
    bias: str,
    preset: str,
    device: torch.device,
    train_ds: BetaReductionDataset,
    val_ds: BetaReductionDataset,
    tokenizer: ProgramTokenizer,
    epochs: int = 5,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    config = ResonanceConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.max_length,
        embed_dim=128,
        n_layers=4,
        n_heads=4,
        ff_dim=512,
        n_frequencies=32,
        dropout=0.1,
        batch_size=16,
        learning_rate=3e-4,
        phase_embedding=phase,
        resonance_kernel=kernel,
        bias_mode=bias,
        init_preset=preset,
        use_phase_stream=True,
        use_resonance_bias=True,
    )
    model = ResonanceTransformer(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    contrastive = VectorizedContrastiveLoss(temperature=0.07)

    def encode_batch(texts: list[str]) -> torch.Tensor:
        tokens = tokenizer.batch_encode(texts, device=str(device))
        # Forward through model to get final hidden states, then mean pool
        with torch.no_grad() if not model.training else torch.enable_grad():
            x, resonance = model.embedding(tokens)
            mask = model.causal_mask[:tokens.size(1), :tokens.size(1)].unsqueeze(0)
            for block in model.blocks:
                x = block(x, resonance, mask)
            x = model.ln_final(x)
            # Mean pool over non-pad
            pad_mask = tokens != tokenizer.pad_id
            x = x * pad_mask.unsqueeze(-1).float()
            lengths = pad_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return x.sum(dim=1) / lengths

    # Training
    history = []
    for epoch in range(epochs):
        model.train()
        import random
        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(indices), config.batch_size):
            batch_idx = indices[i:i + config.batch_size]
            originals = [train_ds.pairs[j]["original"] for j in batch_idx]
            nfs = [train_ds.pairs[j]["normal_form"] for j in batch_idx]

            orig_emb = encode_batch(originals)
            nf_emb = encode_batch(nfs)

            B = orig_emb.size(0)
            all_emb = torch.cat([orig_emb, nf_emb], dim=0)
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

    # Evaluation
    model.eval()
    with torch.no_grad():
        all_origs = [p["original"] for p in val_ds.pairs]
        all_nfs = [p["normal_form"] for p in val_ds.pairs]
        orig_emb = F.normalize(encode_batch(all_origs), dim=-1)
        nf_emb = F.normalize(encode_batch(all_nfs), dim=-1)
        sim = orig_emb @ nf_emb.T

        # Recall@k
        recalls = {}
        for k in [1, 3, 5]:
            _, topk = sim.topk(k, dim=-1)
            correct = sum(1 for i in range(len(val_ds.pairs)) if i in topk[i].cpu().tolist())
            recalls[f"r@{k}"] = correct / len(val_ds.pairs)

        # MRR
        ranks = []
        for i in range(len(val_ds.pairs)):
            order = sim[i].argsort(descending=True)
            rank = (order == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(1.0 / rank)
        mrr = sum(ranks) / len(ranks)

        # Gap
        pos = sim.diagonal().mean().item()
        neg_mask = ~torch.eye(len(val_ds.pairs), dtype=torch.bool, device=device)
        neg = sim[neg_mask].mean().item()

    return {
        "phase": phase,
        "kernel": kernel,
        "bias": bias,
        "preset": preset,
        "params": count_params(model),
        "final_loss": history[-1],
        **recalls,
        "mrr": mrr,
        "pos_sim": pos,
        "neg_sim": neg,
        "gap": pos - neg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_pairs", type=int, default=1000)
    parser.add_argument("--val_pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating beta-reduction datasets...")
    train_ds = BetaReductionDataset(
        n_pairs=args.train_pairs, max_depth=5, max_size=20,
        min_trace_length=2, generator_seed=args.seed,
    )
    val_ds = BetaReductionDataset(
        n_pairs=args.val_pairs, max_depth=5, max_size=20,
        min_trace_length=2, generator_seed=args.seed + 1,
    )
    tokenizer = ProgramTokenizer(max_length=128)
    print(f"  Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    phases = ["real", "fourier_fixed", "complex_angle"]
    kernels = ["cosine", "complex_magnitude", "rbf", "bilinear"]
    biases = ["additive", "residual_gate"]
    presets = ["default", "strong"]

    total = len(phases) * len(kernels) * len(biases) * len(presets)
    print(f"\nRunning {total} conditions...\n")

    results = []
    for phase in phases:
        for kernel in kernels:
            for bias in biases:
                for preset in presets:
                    name = f"{phase}_{kernel}_{bias}_{preset}"
                    print(f"[{len(results)+1}/{total}] {name}")
                    start = time.perf_counter()
                    try:
                        row = run_condition(
                            phase, kernel, bias, preset, device,
                            train_ds, val_ds, tokenizer, epochs=args.epochs, seed=args.seed,
                        )
                        results.append(row)
                        elapsed = time.perf_counter() - start
                        print(f"  → R@1={row['r@1']:.3f} MRR={row['mrr']:.3f} gap={row['gap']:.3f} loss={row['final_loss']:.3f} ({elapsed:.1f}s)")
                    except Exception as e:
                        print(f"  → FAILED: {e}")
                        results.append({"phase": phase, "kernel": kernel, "bias": bias, "preset": preset, "error": str(e)})

    # Save
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print sorted by R@1
    good = [r for r in results if "r@1" in r]
    good.sort(key=lambda x: x["r@1"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 10 BY RECALL@1")
    print("=" * 80)
    print(f"{'Rank':>4} {'Phase':>15} {'Kernel':>18} {'Bias':>16} {'Preset':>8} {'R@1':>6} {'MRR':>6} {'Gap':>6}")
    print("-" * 80)
    for i, r in enumerate(good[:10], 1):
        print(f"{i:>4} {r['phase']:>15} {r['kernel']:>18} {r['bias']:>16} {r['preset']:>8} {r['r@1']:>6.3f} {r['mrr']:>6.3f} {r['gap']:>6.3f}")
    print("=" * 80)

    # Print sorted by gap
    good.sort(key=lambda x: x["gap"], reverse=True)
    print("\nTOP 10 BY POS-NEG GAP")
    print("=" * 80)
    print(f"{'Rank':>4} {'Phase':>15} {'Kernel':>18} {'Bias':>16} {'Preset':>8} {'R@1':>6} {'MRR':>6} {'Gap':>6}")
    print("-" * 80)
    for i, r in enumerate(good[:10], 1):
        print(f"{i:>4} {r['phase']:>15} {r['kernel']:>18} {r['bias']:>16} {r['preset']:>8} {r['r@1']:>6.3f} {r['mrr']:>6.3f} {r['gap']:>6.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
