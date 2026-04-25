#!/usr/bin/env python3
"""
Deliverable 2: Compressibility of Learned Parameters
======================================================
Test quantization, pruning, SVD truncation, and phase-only compression.
Measure compression ratio, validation perplexity, and accuracy @k.

Usage:
    python experiments/param_compressibility.py \
        --standard_path checkpoints/standard.pt \
        --resonance_path checkpoints/resonance.pt \
        --output_json results/compressibility_results.json \
        --output_plot figures/compressibility.png
"""
import argparse
import copy
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import StandardTransformer, ResonanceTransformer
from utils.data import TinyShakespeare, get_shakespeare_text, get_dataloader
from utils.training import compute_perplexity, compute_accuracy_at_k, load_checkpoint, train_model


# ---------------------------------------------------------------------------
# Compression methods
# ---------------------------------------------------------------------------

def quantize_weights(model: nn.Module, n_bits: int) -> dict:
    """
    Simulate INTn quantization by rounding weights to uniform buckets.
    Returns compressed state dict (as float tensors for evaluation).
    """
    n_levels = 2 ** n_bits
    new_state = {}
    for name, param in model.state_dict().items():
        if param.dtype.is_floating_point and param.numel() > 0:
            p_min = param.min()
            p_max = param.max()
            scale = (p_max - p_min) / (n_levels - 1)
            if scale == 0:
                new_state[name] = param.clone()
            else:
                quantized = torch.round((param - p_min) / scale).clamp(0, n_levels - 1)
                dequantized = quantized * scale + p_min
                new_state[name] = dequantized
        else:
            new_state[name] = param.clone()
    return new_state


def prune_weights(model: nn.Module, sparsity: float, global_prune: bool = True) -> dict:
    """
    Magnitude-based pruning. Zero out smallest-magnitude weights.
    Returns state dict with pruned weights.
    """
    new_state = {}
    if global_prune:
        # Collect all weights
        all_weights = []
        for param in model.parameters():
            if param.dim() > 1:
                all_weights.append(param.abs().flatten())
        if len(all_weights) == 0:
            return {k: v.clone() for k, v in model.state_dict().items()}
        all_weights = torch.cat(all_weights)
        k = int(sparsity * all_weights.numel())
        if k > 0:
            threshold = torch.kthvalue(all_weights, k).values.item()
        else:
            threshold = float('inf')
        for name, param in model.state_dict().items():
            if param.dim() > 1 and param.dtype.is_floating_point:
                mask = param.abs() >= threshold
                new_state[name] = param * mask
            else:
                new_state[name] = param.clone()
    else:
        # Per-layer pruning
        for name, param in model.state_dict().items():
            if param.dim() > 1 and param.dtype.is_floating_point:
                flat = param.abs().flatten()
                k = int(sparsity * flat.numel())
                if k > 0:
                    thr = torch.kthvalue(flat, k).values.item()
                    mask = param.abs() >= thr
                    new_state[name] = param * mask
                else:
                    new_state[name] = param.clone()
            else:
                new_state[name] = param.clone()
    return new_state


def svd_truncate_embeddings(model, k_ratio: float, semantic_only: bool = False) -> dict:
    """
    SVD truncation for embedding matrices.
    Keep top-k singular values where k = k_ratio * min(M, N).
    """
    new_state = {name: param.clone() for name, param in model.state_dict().items()}
    emb_names = []
    if hasattr(model, "semantic_embedding"):
        emb_names.append("semantic_embedding.weight")
    if hasattr(model, "embedding"):
        emb_names.append("embedding.weight")
    if not semantic_only and hasattr(model, "phase_embedding"):
        emb_names.append("phase_embedding.weight")

    for name in emb_names:
        if name not in new_state:
            continue
        W = new_state[name]
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        rank = min(W.shape)
        k = max(1, int(k_ratio * rank))
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        W_approx = U_k @ torch.diag(S_k) @ Vt_k
        new_state[name] = W_approx
    return new_state


def compress_phase_more_aggressively(model: ResonanceTransformer, phase_bits: int, sem_bits: int) -> dict:
    """
    Quantize phase embeddings more aggressively than semantic.
    phase_bits < sem_bits (e.g., phase_bits=4, sem_bits=8).
    """
    new_state = {}
    for name, param in model.state_dict().items():
        new_state[name] = param.clone()

    # Quantize phase embeddings
    if "phase_embedding.weight" in new_state:
        p = new_state["phase_embedding.weight"]
        n_levels = 2 ** phase_bits
        p_min, p_max = p.min(), p.max()
        scale = (p_max - p_min) / (n_levels - 1)
        if scale > 0:
            q = torch.round((p - p_min) / scale).clamp(0, n_levels - 1)
            new_state["phase_embedding.weight"] = q * scale + p_min

    # Quantize semantic embeddings
    if "semantic_embedding.weight" in new_state:
        p = new_state["semantic_embedding.weight"]
        n_levels = 2 ** sem_bits
        p_min, p_max = p.min(), p.max()
        scale = (p_max - p_min) / (n_levels - 1)
        if scale > 0:
            q = torch.round((p - p_min) / scale).clamp(0, n_levels - 1)
            new_state["semantic_embedding.weight"] = q * scale + p_min

    return new_state


def compress_semantic_only(model, sem_bits: int, phase_bits: int) -> dict:
    """Quantize semantic more aggressively than phase."""
    return compress_phase_more_aggressively(model, phase_bits, sem_bits)


# ---------------------------------------------------------------------------
# Compression ratio estimation
# ---------------------------------------------------------------------------

def estimate_compression_ratio(model: nn.Module, compressed_state: dict, method: str) -> float:
    """
    Estimate compression ratio (original bytes / compressed bytes).
    For quantization: assume float32 -> n_bits per element.
    For pruning: count non-zero elements.
    For SVD: store U_k, S_k, Vt_k.
    """
    original_bytes = sum(p.numel() * 4 for p in model.parameters())  # float32

    if method.startswith("quant"):
        bits = int(method.split("_")[-1].replace("int", ""))
        # All parameters quantized to n_bits (but we store as float for eval)
        # For ratio estimation, assume packed storage
        compressed_bytes = sum(p.numel() * bits / 8 for p in model.parameters())
    elif method.startswith("prune"):
        # Count nonzeros in compressed state
        nonzero = 0
        for p in compressed_state.values():
            if p.dtype.is_floating_point:
                nonzero += (p != 0).sum().item()
        # Sparse matrix: store (index, value) pairs, 4 bytes each
        compressed_bytes = nonzero * 8  # index + value
    elif method.startswith("svd"):
        # Embedding matrices only; everything else full size
        compressed_bytes = original_bytes
        emb_name = None
        if hasattr(model, "semantic_embedding"):
            emb_name = "semantic_embedding.weight"
        elif hasattr(model, "embedding"):
            emb_name = "embedding.weight"
        if emb_name and emb_name in compressed_state:
            W = compressed_state[emb_name]
            rank_approx = torch.linalg.matrix_rank(W, tol=1e-3).item()
            M, N = W.shape
            # U_k (M x k), S_k (k), Vt_k (k x N)
            svd_bytes = (M * rank_approx + rank_approx + rank_approx * N) * 4
            # Replace original embedding bytes with SVD bytes
            compressed_bytes -= (M * N * 4)
            compressed_bytes += svd_bytes
    elif method.startswith("phase_quant") or method.startswith("sem_quant"):
        # Mixed quantization: estimate average bits
        if hasattr(model, "phase_embedding") and hasattr(model, "semantic_embedding"):
            phase_el = model.phase_embedding.weight.numel()
            sem_el = model.semantic_embedding.weight.numel()
            other_el = sum(p.numel() for n, p in model.named_parameters()
                          if "embedding" not in n)
            # Parse bits from method name like "phase_quant_4_8"
            parts = method.split("_")
            pb = int(parts[2])
            sb = int(parts[3])
            compressed_bytes = (phase_el * pb / 8) + (sem_el * sb / 8) + (other_el * 4)
        else:
            compressed_bytes = original_bytes
    else:
        compressed_bytes = original_bytes

    if compressed_bytes == 0:
        return 1.0
    return original_bytes / compressed_bytes


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def evaluate_compressed_model(model: nn.Module, compressed_state: dict, val_loader, device: str):
    """Load compressed state and evaluate."""
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(compressed_state, strict=False)
    ppl = compute_perplexity(model, val_loader, device=device, max_batches=30)
    acc1 = compute_accuracy_at_k(model, val_loader, k=1, device=device, max_batches=30)
    acc5 = compute_accuracy_at_k(model, val_loader, k=5, device=device, max_batches=30)
    # Restore original
    model.load_state_dict(original_state)
    return ppl, acc1, acc5


def run_compressibility_experiment(model: nn.Module, model_name: str, val_loader, device: str) -> list:
    """Run all compression experiments on a single model."""
    results = []

    # 1. Quantization
    for bits in [8, 4]:
        method = f"quant_int{bits}"
        cstate = quantize_weights(model, bits)
        ratio = estimate_compression_ratio(model, cstate, method)
        ppl, acc1, acc5 = evaluate_compressed_model(model, cstate, val_loader, device)
        results.append({
            "model": model_name,
            "method": method,
            "compression_ratio": ratio,
            "ppl": ppl,
            "acc@1": acc1,
            "acc@5": acc5,
        })
        print(f"  {method}: ratio={ratio:.2f}x ppl={ppl:.2f} acc@1={acc1:.4f}")

    # 2. Pruning
    for sparsity in [0.1, 0.3, 0.5, 0.7, 0.9]:
        method = f"prune_{sparsity}"
        cstate = prune_weights(model, sparsity)
        ratio = estimate_compression_ratio(model, cstate, method)
        ppl, acc1, acc5 = evaluate_compressed_model(model, cstate, val_loader, device)
        results.append({
            "model": model_name,
            "method": method,
            "compression_ratio": ratio,
            "ppl": ppl,
            "acc@1": acc1,
            "acc@5": acc5,
        })
        print(f"  {method}: ratio={ratio:.2f}x ppl={ppl:.2f} acc@1={acc1:.4f}")

    # 3. SVD truncation on embeddings
    for k_ratio in [0.9, 0.7, 0.5, 0.3, 0.1]:
        method = f"svd_{k_ratio}"
        cstate = svd_truncate_embeddings(model, k_ratio, semantic_only=False)
        ratio = estimate_compression_ratio(model, cstate, method)
        ppl, acc1, acc5 = evaluate_compressed_model(model, cstate, val_loader, device)
        results.append({
            "model": model_name,
            "method": method,
            "compression_ratio": ratio,
            "ppl": ppl,
            "acc@1": acc1,
            "acc@5": acc5,
        })
        print(f"  {method}: ratio={ratio:.2f}x ppl={ppl:.2f} acc@1={acc1:.4f}")

    # 4. Phase-only aggressive compression (Resonance only)
    if isinstance(model, ResonanceTransformer):
        for pb, sb in [(4, 8), (2, 8), (4, 16), (2, 16)]:
            method = f"phase_quant_{pb}_{sb}"
            cstate = compress_phase_more_aggressively(model, pb, sb)
            ratio = estimate_compression_ratio(model, cstate, method)
            ppl, acc1, acc5 = evaluate_compressed_model(model, cstate, val_loader, device)
            results.append({
                "model": model_name,
                "method": method,
                "compression_ratio": ratio,
                "ppl": ppl,
                "acc@1": acc1,
                "acc@5": acc5,
            })
            print(f"  {method}: ratio={ratio:.2f}x ppl={ppl:.2f} acc@1={acc1:.4f}")

        # 5. Semantic-only aggressive compression
        for sb, pb in [(4, 8), (2, 8)]:
            method = f"sem_quant_{sb}_{pb}"
            cstate = compress_semantic_only(model, sb, pb)
            ratio = estimate_compression_ratio(model, cstate, method)
            ppl, acc1, acc5 = evaluate_compressed_model(model, cstate, val_loader, device)
            results.append({
                "model": model_name,
                "method": method,
                "compression_ratio": ratio,
                "ppl": ppl,
                "acc@1": acc1,
                "acc@5": acc5,
            })
            print(f"  {method}: ratio={ratio:.2f}x ppl={ppl:.2f} acc@1={acc1:.4f}")

    return results


def plot_compressibility(results: list, save_path: str):
    """Plot compression ratio vs perplexity and accuracy."""
    # Group by model
    std_results = [r for r in results if r["model"] == "standard"]
    res_results = [r for r in results if r["model"] == "resonance"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = ["quant_int8", "quant_int4", "prune_0.1", "prune_0.3", "prune_0.5", "prune_0.7", "prune_0.9",
                 "svd_0.9", "svd_0.7", "svd_0.5", "svd_0.3", "svd_0.1"]
    method_labels = {m: m.replace("_", " ") for m in methods}

    def plot_scatter(ax, results_list, label, marker):
        x = [r["compression_ratio"] for r in results_list]
        y = [r["ppl"] for r in results_list]
        ax.scatter(x, y, label=label, marker=marker, s=80, alpha=0.7)
        for r in results_list:
            ax.annotate(r["method"].replace("_", " "), (r["compression_ratio"], r["ppl"]), fontsize=6, alpha=0.6)

    ax = axes[0]
    if std_results:
        plot_scatter(ax, std_results, "Standard", "o")
    if res_results:
        plot_scatter(ax, res_results, "Resonance", "s")
    ax.set_xlabel("Compression Ratio (higher = more compressed)")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title("Compression vs Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if std_results:
        x = [r["compression_ratio"] for r in std_results]
        y = [r["acc@1"] for r in std_results]
        ax.scatter(x, y, label="Standard", marker="o", s=80, alpha=0.7)
    if res_results:
        x = [r["compression_ratio"] for r in res_results]
        y = [r["acc@1"] for r in res_results]
        ax.scatter(x, y, label="Resonance", marker="s", s=80, alpha=0.7)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Accuracy @ 1")
    ax.set_title("Compression vs Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Compressibility] Plot saved to {save_path}")
    plt.close()


def build_tiny_model(model_type: str, vocab_size: int, device: str):
    """Build a tiny model for smoke tests."""
    if model_type == "standard":
        model = StandardTransformer(
            vocab_size=vocab_size, d_model=128, nhead=4, num_layers=3,
            dim_feedforward=512, dropout=0.1, max_seq_len=128,
        )
    else:
        model = ResonanceTransformer(
            vocab_size=vocab_size, d_model=128, nhead=4, num_layers=3,
            dim_feedforward=512, dropout=0.1, max_seq_len=128,
            n_frequencies=32, use_resonance=True,
        )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Parameter Compressibility Experiment")
    parser.add_argument("--standard_path", type=str, default=None)
    parser.add_argument("--resonance_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, default="results/compressibility_results.json")
    parser.add_argument("--output_plot", type=str, default="figures/compressibility.png")
    parser.add_argument("--train_if_missing", action="store_true", default=True)
    args = parser.parse_args()

    device = args.device
    print(f"[Compressibility] Using device: {device}")

    # Data
    text = get_shakespeare_text()
    dataset = TinyShakespeare(text, seq_len=128)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    vocab_size = dataset.vocab_size
    print(f"[Compressibility] Vocab={vocab_size}, train={n_train}, val={n_val}")

    all_results = []

    for model_type, ckpt_path in [("standard", args.standard_path), ("resonance", args.resonance_path)]:
        print(f"\n[Compressibility] ===== {model_type.upper()} =====")
        model = build_tiny_model(model_type, vocab_size, device)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[Compressibility] Loading {ckpt_path}")
            load_checkpoint(model, ckpt_path, device=device)
        elif args.train_if_missing:
            print(f"[Compressibility] No checkpoint. Training {model_type}...")
            train_model(model, train_loader, val_loader, device=device, num_epochs=5, lr=3e-4,
                        checkpoint_dir="checkpoints", model_name=model_type, log_interval=10)
            ckpt_path = f"checkpoints/{model_type}.pt"
        else:
            print(f"[Compressibility] Using random init for {model_type}")

        results = run_compressibility_experiment(model, model_type, val_loader, device)
        all_results.extend(results)

    # Save JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Compressibility] Results saved to {args.output_json}")

    # Plot
    plot_compressibility(all_results, args.output_plot)
    print("[Compressibility] Done.")


if __name__ == "__main__":
    main()
