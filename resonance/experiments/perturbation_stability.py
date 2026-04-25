#!/usr/bin/env python3
"""
Deliverable 1: Perplexity Stability Under Perturbation
=========================================================
Apply Gaussian noise to different components of StandardTransformer
and ResonanceTransformer, measure validation perplexity, compute
stability ratio (ppl_perturbed / ppl_clean).

Usage:
    python experiments/perturbation_stability.py \
        --model_path checkpoints/resonance.pt \
        --model_type resonance \
        --data_path data/ \
        --output_json results/perturbation_results.json \
        --output_plot figures/perturbation_stability.png
"""
import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import StandardTransformer, ResonanceTransformer
from utils.data import TinyShakespeare, get_shakespeare_text, get_dataloader
from utils.training import compute_perplexity, load_checkpoint, train_model


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def perturb_tensor(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise with std sigma * (tensor.std() or 1.0 if std==0)."""
    std = tensor.std(unbiased=False).item()
    if std == 0 or math.isnan(std):
        std = 1.0
    noise = torch.randn_like(tensor) * sigma * std
    return tensor + noise


def perturb_phase_embeddings_only(model: ResonanceTransformer, sigma: float):
    """Perturb only E_p (phase embeddings)."""
    with torch.no_grad():
        model.phase_embedding.weight.copy_(perturb_tensor(model.phase_embedding.weight, sigma))


def perturb_semantic_embeddings_only(model, sigma: float):
    """Perturb only E_s (semantic embeddings)."""
    emb_name = "semantic_embedding" if hasattr(model, "semantic_embedding") else "embedding"
    with torch.no_grad():
        getattr(model, emb_name).weight.copy_(perturb_tensor(getattr(model, emb_name).weight, sigma))


def perturb_full_weights(model: nn.Module, sigma: float):
    """Perturb all parameters (excluding buffers)."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(perturb_tensor(p, sigma))


def perturb_alpha_only(model: ResonanceTransformer, sigma: float):
    """Perturb only the blend parameter alpha."""
    with torch.no_grad():
        model.alpha.copy_(perturb_tensor(model.alpha, sigma))


def perturb_resonance_weight_only(model: ResonanceTransformer, sigma: float):
    """Perturb only the resonance attention scalar weight."""
    with torch.no_grad():
        model.resonance_weight.copy_(perturb_tensor(model.resonance_weight, sigma))


PERTURBATION_REGISTRY = {
    "phase_embeddings": perturb_phase_embeddings_only,
    "semantic_embeddings": perturb_semantic_embeddings_only,
    "full_weights": perturb_full_weights,
    "alpha_blend": perturb_alpha_only,
    "resonance_weight": perturb_resonance_weight_only,
}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_perturbation_experiment(
    model: nn.Module,
    model_type: str,
    val_loader,
    sigmas: list,
    perturbation_types: list,
    device: str,
) -> dict:
    """
    Run the full perturbation experiment.
    Returns dict with clean_ppl and per-perturbation results.
    """
    # Save original state
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Clean perplexity
    print("[Perturbation] Computing clean perplexity...")
    clean_ppl = compute_perplexity(model, val_loader, device=device, max_batches=30)
    print(f"[Perturbation] Clean PPL = {clean_ppl:.4f}")

    results = {
        "model_type": model_type,
        "clean_ppl": clean_ppl,
        "sigmas": sigmas,
        "perturbations": {},
    }

    for ptype in perturbation_types:
        if model_type == "standard" and ptype in ("alpha_blend", "resonance_weight", "phase_embeddings"):
            print(f"[Perturbation] Skipping '{ptype}' for standard model (not applicable).")
            continue

        print(f"[Perturbation] Running {ptype}...")
        ppl_list = []
        ratio_list = []
        for sigma in sigmas:
            # Restore original state
            model.load_state_dict(original_state)

            # Apply perturbation
            if ptype == "phase_embeddings":
                perturb_phase_embeddings_only(model, sigma)
            elif ptype == "semantic_embeddings":
                perturb_semantic_embeddings_only(model, sigma)
            elif ptype == "full_weights":
                perturb_full_weights(model, sigma)
            elif ptype == "alpha_blend":
                perturb_alpha_only(model, sigma)
            elif ptype == "resonance_weight":
                perturb_resonance_weight_only(model, sigma)
            else:
                raise ValueError(f"Unknown perturbation type: {ptype}")

            ppl = compute_perplexity(model, val_loader, device=device, max_batches=30)
            ratio = ppl / clean_ppl
            ppl_list.append(float(ppl))
            ratio_list.append(float(ratio))
            print(f"  sigma={sigma:.4f} -> ppl={ppl:.4f} ratio={ratio:.4f}")

        results["perturbations"][ptype] = {
            "ppl": ppl_list,
            "stability_ratio": ratio_list,
        }

    # Restore clean state
    model.load_state_dict(original_state)
    return results


def plot_stability_curves(results: dict, save_path: str):
    """Generate stability curves plot."""
    sigmas = results["sigmas"]
    perturbations = results["perturbations"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute PPL
    ax = axes[0]
    for ptype, data in perturbations.items():
        ax.plot(sigmas, data["ppl"], marker='o', label=ptype.replace('_', ' '))
    ax.axhline(results["clean_ppl"], color='black', linestyle='--', label='clean ppl')
    ax.set_xlabel('Perturbation scale σ')
    ax.set_ylabel('Validation Perplexity')
    ax.set_title(f'Perturbed PPL ({results["model_type"]})')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: stability ratio
    ax = axes[1]
    for ptype, data in perturbations.items():
        ax.plot(sigmas, data["stability_ratio"], marker='o', label=ptype.replace('_', ' '))
    ax.axhline(1.0, color='black', linestyle='--', label='baseline')
    ax.set_xlabel('Perturbation scale σ')
    ax.set_ylabel('Stability Ratio (ppl_perturbed / ppl_clean)')
    ax.set_title(f'Stability Ratio ({results["model_type"]})')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Perturbation] Plot saved to {save_path}")
    plt.close()


def build_tiny_model(model_type: str, vocab_size: int, device: str):
    """Build a tiny model for smoke tests."""
    if model_type == "standard":
        model = StandardTransformer(
            vocab_size=vocab_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            max_seq_len=128,
        )
    else:
        model = ResonanceTransformer(
            vocab_size=vocab_size,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            max_seq_len=128,
            n_frequencies=32,
            use_resonance=True,
        )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Perturbation Stability Experiment")
    parser.add_argument("--model_path", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--model_type", type=str, choices=["standard", "resonance"], default="resonance")
    parser.add_argument("--data_path", type=str, default=None, help="Data directory (unused for built-in dataset)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, default="results/perturbation_results.json")
    parser.add_argument("--output_plot", type=str, default="figures/perturbation_stability.png")
    parser.add_argument("--train_if_missing", action="store_true", default=True)
    args = parser.parse_args()

    device = args.device
    print(f"[Perturbation] Using device: {device}")

    # Load / create data
    text = get_shakespeare_text()
    dataset = TinyShakespeare(text, seq_len=128)
    # 80/20 split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    vocab_size = dataset.vocab_size
    print(f"[Perturbation] Vocab size={vocab_size}, train={n_train}, val={n_val}")

    # Load or train model
    model = build_tiny_model(args.model_type, vocab_size, device)
    if args.model_path and os.path.exists(args.model_path):
        print(f"[Perturbation] Loading checkpoint from {args.model_path}")
        load_checkpoint(model, args.model_path, device=device)
    elif args.train_if_missing:
        print("[Perturbation] No checkpoint found. Training tiny model first...")
        train_model(
            model, train_loader, val_loader,
            device=device, num_epochs=5, lr=3e-4,
            checkpoint_dir="checkpoints",
            model_name=args.model_type,
            log_interval=10,
        )
        # Save checkpoint path for next time
        args.model_path = f"checkpoints/{args.model_type}.pt"
    else:
        print("[Perturbation] No checkpoint and train_if_missing=False. Using random init.")

    # Run experiment
    sigmas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    perturbation_types = [
        "phase_embeddings",
        "semantic_embeddings",
        "full_weights",
        "alpha_blend",
        "resonance_weight",
    ]
    if args.model_type == "standard":
        perturbation_types = ["semantic_embeddings", "full_weights"]

    results = run_perturbation_experiment(
        model, args.model_type, val_loader, sigmas, perturbation_types, device
    )

    # Save JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Perturbation] Results saved to {args.output_json}")

    # Plot
    plot_stability_curves(results, args.output_plot)
    print("[Perturbation] Done.")


if __name__ == "__main__":
    main()
