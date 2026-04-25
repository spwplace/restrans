#!/usr/bin/env python3
"""
Deliverable 3: Compressibility of "Gestalt Streams"
===================================================
Extract semantic stream, phase stream, resonance matrices, and blend values.
Measure mutual information, PCA dimensionality, cross-predictability,
resonance entropy, effective rank, and independent compressibility.

Usage:
    python experiments/gestalt_compressibility.py \
        --model_path checkpoints/resonance.pt \
        --output_json results/gestalt_results.json \
        --output_plot figures/gestalt_analysis.png
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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ResonanceTransformer
from utils.data import TinyShakespeare, get_shakespeare_text, get_dataloader
from utils.training import compute_perplexity, load_checkpoint, train_model


# ---------------------------------------------------------------------------
# Stream extraction
# ---------------------------------------------------------------------------

def extract_streams(model: ResonanceTransformer, tokens: torch.Tensor, device: str):
    """
    Extract semantic stream, phase stream, resonance matrices, and blend.
    Returns dict with:
        h_sem: [B, S, D]
        h_phase: [B, S, D]
        R: [B, S, S]
        alpha: [D]
    """
    model.eval()
    with torch.no_grad():
        tokens = tokens.to(device)
        h_sem = model.semantic_embedding(tokens)           # [B, S, D]
        h_phase_raw = model.phase_embedding(tokens)        # [B, S, F]
        h_phase = model.phase_proj(h_phase_raw)           # [B, S, D]
        R = model.compute_resonance_matrix(tokens)          # [B, S, S]
        alpha = torch.sigmoid(model.alpha).cpu().numpy()    # [D]

    return {
        "h_sem": h_sem.cpu().numpy(),
        "h_phase": h_phase.cpu().numpy(),
        "R": R.cpu().numpy(),
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def mutual_information_knn(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """
    Simple k-NN based mutual information estimator.
    X, Y are [N, D] arrays. We bin each into discrete bins and compute MI.
    """
    # Simple binning approach for MI estimation
    def discretize(Z, n_bins=20):
        Z_min, Z_max = Z.min(axis=0), Z.max(axis=0)
        # Avoid division by zero
        delta = Z_max - Z_min
        delta[delta == 0] = 1.0
        bins = np.floor((Z - Z_min) / delta * n_bins).clip(0, n_bins - 1).astype(int)
        # Hash to single integer per sample
        coeffs = np.array([n_bins ** i for i in range(Z.shape[1])])
        coeffs = coeffs[:Z.shape[1]]
        return bins @ coeffs

    # Use PCA to reduce to 10D before discretization to avoid explosion
    if X.shape[1] > 10:
        pca_x = PCA(n_components=10)
        X_red = pca_x.fit_transform(X)
    else:
        X_red = X
    if Y.shape[1] > 10:
        pca_y = PCA(n_components=10)
        Y_red = pca_y.fit_transform(Y)
    else:
        Y_red = Y

    X_d = discretize(X_red, n_bins=15)
    Y_d = discretize(Y_red, n_bins=15)

    # Joint and marginal distributions
    N = len(X_d)
    # Build joint histogram
    joint = {}
    marg_x = {}
    marg_y = {}
    for x, y in zip(X_d, Y_d):
        joint[(x, y)] = joint.get((x, y), 0) + 1
        marg_x[x] = marg_x.get(x, 0) + 1
        marg_y[y] = marg_y.get(y, 0) + 1

    mi = 0.0
    for (x, y), count_xy in joint.items():
        p_xy = count_xy / N
        p_x = marg_x[x] / N
        p_y = marg_y[y] / N
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi


def pca_dimensionality(X: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Number of PCs needed to capture variance_threshold fraction of variance."""
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n = np.searchsorted(cumsum, variance_threshold) + 1
    return int(n)


def cross_predictability(X: np.ndarray, Y: np.ndarray, test_ratio: float = 0.2) -> dict:
    """
    Fit Ridge regression to predict Y from X.
    Returns R^2 score.
    """
    n_test = int(len(X) * test_ratio)
    X_train, X_test = X[:-n_test], X[-n_test:]
    Y_train, Y_test = Y[:-n_test], Y[-n_test:]

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    mse = np.mean((Y_test - Y_pred) ** 2)
    return {"r2": float(r2), "mse": float(mse)}


def matrix_entropy(R: np.ndarray) -> float:
    """
    Shannon entropy of the resonance attention patterns.
    R is [B, S, S]. Treat each row as a distribution after softmax.
    """
    # Apply softmax across last dim
    R_tensor = torch.tensor(R, dtype=torch.float32)
    probs = F.softmax(R_tensor, dim=-1).numpy()
    # Entropy per position, averaged
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
    return float(entropy.mean())


def effective_rank(M: np.ndarray) -> float:
    """
    Effective rank of a matrix: exp of Shannon entropy of singular value distribution.
    """
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    s_norm = s / s.sum()
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    return float(np.exp(entropy))


def independent_compressibility_experiment(model: ResonanceTransformer, val_loader, device: str) -> list:
    """
    Quantize phase embeddings aggressively while keeping semantic full precision, and vice versa.
    """
    results = []
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    def quantize_param(param, bits):
        n_levels = 2 ** bits
        p_min, p_max = param.min(), param.max()
        scale = (p_max - p_min) / (n_levels - 1)
        if scale == 0:
            return param.clone()
        q = torch.round((param - p_min) / scale).clamp(0, n_levels - 1)
        return q * scale + p_min

    configs = [
        ("phase_int4_sem_fp32", {"phase": 4, "semantic": 32}),
        ("phase_int2_sem_fp32", {"phase": 2, "semantic": 32}),
        ("phase_int1_sem_fp32", {"phase": 1, "semantic": 32}),
        ("sem_int4_phase_fp32", {"semantic": 4, "phase": 32}),
        ("sem_int2_phase_fp32", {"semantic": 2, "phase": 32}),
        ("sem_int1_phase_fp32", {"semantic": 1, "phase": 32}),
    ]

    for name, cfg in configs:
        state = {k: v.clone() for k, v in original_state.items()}
        if "phase" in cfg and cfg["phase"] < 32:
            state["phase_embedding.weight"] = quantize_param(state["phase_embedding.weight"], cfg["phase"])
        if "semantic" in cfg and cfg["semantic"] < 32:
            state["semantic_embedding.weight"] = quantize_param(state["semantic_embedding.weight"], cfg["semantic"])
        model.load_state_dict(state)
        ppl = compute_perplexity(model, val_loader, device=device, max_batches=30)
        results.append({"config": name, "ppl": float(ppl), "bits": cfg})
        print(f"  {name}: ppl={ppl:.2f}")

    # Restore
    model.load_state_dict(original_state)
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_gestalt_analysis(streams: dict, model: ResonanceTransformer, save_path: str, dataset=None):
    """Create multi-panel gestalt analysis figure."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    h_sem = streams["h_sem"]
    h_phase = streams["h_phase"]
    R = streams["R"]
    alpha = streams["alpha"]

    # Flatten batch and sequence dims
    N = h_sem.shape[0] * h_sem.shape[1]
    h_sem_flat = h_sem.reshape(N, -1)
    h_phase_flat = h_phase.reshape(N, -1)

    # 1. PCA scree plots
    ax1 = fig.add_subplot(gs[0, 0])
    pca_sem = PCA()
    pca_sem.fit(h_sem_flat)
    ax1.plot(np.cumsum(pca_sem.explained_variance_ratio_), marker='o', label='Semantic')
    pca_phase = PCA()
    pca_phase.fit(h_phase_flat)
    ax1.plot(np.cumsum(pca_phase.explained_variance_ratio_), marker='s', label='Phase')
    ax1.axhline(0.95, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Number of PCs")
    ax1.set_ylabel("Cumulative Variance")
    ax1.set_title("PCA Dimensionality")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Alpha (blend parameter) distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(alpha)), alpha, color='steelblue')
    ax2.set_xlabel("Dimension")
    ax2.set_ylabel("Blend α (sigmoid)")
    ax2.set_title("Learned Blend Parameter α")
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # 3. Resonance matrix heatmap (first sample)
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(R[0], cmap='viridis', aspect='auto')
    ax3.set_title("Resonance Matrix R (sample 0)")
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # 4. t-SNE of phase embeddings (by first character of token, if available)
    ax4 = fig.add_subplot(gs[1, :2])
    # Phase embedding matrix [V, F]
    phase_emb = model.phase_embedding.weight.detach().cpu().numpy()
    V, F = phase_emb.shape
    if V > 10:
        # t-SNE on phase embeddings
        tsne = TSNE(n_components=2, perplexity=min(30, V-1), random_state=42, max_iter=1000)
        phase_2d = tsne.fit_transform(phase_emb)

        # Color by first character if dataset has char mapping
        if dataset is not None and hasattr(dataset, 'itos'):
            chars = [dataset.itos.get(i, '?')[0] if dataset.itos.get(i, '') else '?' for i in range(V)]
            # Map first char to color using ord
            colors = [ord(c) % 20 if len(c) > 0 else 0 for c in chars]
            scatter = ax4.scatter(phase_2d[:, 0], phase_2d[:, 1], c=colors, cmap='tab20', s=20, alpha=0.7)
            ax4.set_title("t-SNE of Phase Embeddings (colored by first char)")
        else:
            ax4.scatter(phase_2d[:, 0], phase_2d[:, 1], s=20, alpha=0.7)
            ax4.set_title("t-SNE of Phase Embeddings")
    else:
        ax4.text(0.5, 0.5, "Vocab too small for t-SNE", ha='center', va='center')
    ax4.grid(True, alpha=0.3)

    # 5. Effective rank of resonance matrices across batch
    ax5 = fig.add_subplot(gs[1, 2])
    eff_ranks = [effective_rank(R[b]) for b in range(R.shape[0])]
    ax5.bar(range(len(eff_ranks)), eff_ranks, color='coral')
    ax5.set_xlabel("Batch index")
    ax5.set_ylabel("Effective Rank")
    ax5.set_title("Effective Rank of R per sample")
    ax5.grid(True, alpha=0.3)

    # 6. Stream correlation (sample dimensions)
    ax6 = fig.add_subplot(gs[2, 0])
    # Random sample of dimensions
    n_show = min(50, h_sem_flat.shape[1])
    idx = np.random.choice(h_sem_flat.shape[1], n_show, replace=False)
    corr = np.corrcoef(h_sem_flat[:, idx].T, h_phase_flat[:, idx].T)[:n_show, n_show:]
    im2 = ax6.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax6.set_title("Semantic-Phase Cross-Correlation")
    ax6.set_xlabel("Phase dims")
    ax6.set_ylabel("Semantic dims")
    plt.colorbar(im2, ax=ax6, fraction=0.046)

    # 7. Resonance entropy distribution
    ax7 = fig.add_subplot(gs[2, 1])
    entropies = []
    for b in range(R.shape[0]):
        probs = np.exp(R[b]) / np.sum(np.exp(R[b]), axis=-1, keepdims=True)
        eps = 1e-12
        ent = -np.sum(probs * np.log(probs + eps), axis=-1)
        entropies.extend(ent.tolist())
    ax7.hist(entropies, bins=30, color='seagreen', edgecolor='black')
    ax7.set_xlabel("Resonance Entropy")
    ax7.set_ylabel("Frequency")
    ax7.set_title("Distribution of R Entropy")
    ax7.grid(True, alpha=0.3)

    # 8. Stream norm comparison
    ax8 = fig.add_subplot(gs[2, 2])
    sem_norms = np.linalg.norm(h_sem_flat, axis=1)
    phase_norms = np.linalg.norm(h_phase_flat, axis=1)
    ax8.hist(sem_norms, bins=30, alpha=0.5, label='Semantic', color='blue')
    ax8.hist(phase_norms, bins=30, alpha=0.5, label='Phase', color='orange')
    ax8.set_xlabel("L2 Norm")
    ax8.set_ylabel("Frequency")
    ax8.set_title("Stream Norm Distribution")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Gestalt] Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_tiny_resonance(vocab_size: int, device: str):
    return ResonanceTransformer(
        vocab_size=vocab_size, d_model=128, nhead=4, num_layers=3,
        dim_feedforward=512, dropout=0.1, max_seq_len=128,
        n_frequencies=32, use_resonance=True,
    ).to(device)


def main():
    parser = argparse.ArgumentParser(description="Gestalt Stream Compressibility Experiment")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, default="results/gestalt_results.json")
    parser.add_argument("--output_plot", type=str, default="figures/gestalt_analysis.png")
    parser.add_argument("--train_if_missing", action="store_true", default=True)
    args = parser.parse_args()

    device = args.device
    print(f"[Gestalt] Using device: {device}")

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
    print(f"[Gestalt] Vocab={vocab_size}, train={n_train}, val={n_val}")

    # Load / train model
    model = build_tiny_resonance(vocab_size, device)
    if args.model_path and os.path.exists(args.model_path):
        print(f"[Gestalt] Loading {args.model_path}")
        load_checkpoint(model, args.model_path, device=device)
    elif args.train_if_missing:
        print("[Gestalt] No checkpoint. Training tiny ResonanceTransformer...")
        train_model(model, train_loader, val_loader, device=device, num_epochs=5, lr=3e-4,
                    checkpoint_dir="checkpoints", model_name="resonance", log_interval=10)
        args.model_path = "checkpoints/resonance.pt"
    else:
        print("[Gestalt] Using random init.")

    # Extract streams from a validation batch
    print("[Gestalt] Extracting streams from validation batch...")
    val_batch = next(iter(val_loader))
    tokens = val_batch[0]  # [B, S]
    streams = extract_streams(model, tokens, device)

    # Flatten for analysis
    N = streams["h_sem"].shape[0] * streams["h_sem"].shape[1]
    h_sem_flat = streams["h_sem"].reshape(N, -1)
    h_phase_flat = streams["h_phase"].reshape(N, -1)
    R_all = streams["R"].reshape(-1, streams["R"].shape[-1])

    print("[Gestalt] Computing mutual information...")
    # Sample for MI (too expensive on full)
    sample_size = min(2000, N)
    idx = np.random.choice(N, sample_size, replace=False)
    mi = mutual_information_knn(h_sem_flat[idx], h_phase_flat[idx], k=5)
    print(f"  MI(semantic, phase) ≈ {mi:.4f} bits")

    print("[Gestalt] Computing PCA dimensionality...")
    pca_sem_dim = pca_dimensionality(h_sem_flat, 0.95)
    pca_phase_dim = pca_dimensionality(h_phase_flat, 0.95)
    print(f"  Semantic 95% variance: {pca_sem_dim} dims")
    print(f"  Phase 95% variance: {pca_phase_dim} dims")

    print("[Gestalt] Computing cross-predictability...")
    cp = cross_predictability(h_phase_flat, h_sem_flat, test_ratio=0.2)
    print(f"  Phase -> Semantic: R^2={cp['r2']:.4f} MSE={cp['mse']:.4f}")
    cp_rev = cross_predictability(h_sem_flat, h_phase_flat, test_ratio=0.2)
    print(f"  Semantic -> Phase: R^2={cp_rev['r2']:.4f} MSE={cp_rev['mse']:.4f}")

    print("[Gestalt] Computing resonance entropy...")
    r_entropy = matrix_entropy(streams["R"])
    print(f"  Mean resonance entropy: {r_entropy:.4f} nats")

    print("[Gestalt] Computing effective rank of R...")
    eff_r = effective_rank(R_all)
    print(f"  Effective rank: {eff_r:.2f}")

    print("[Gestalt] Running independent compressibility...")
    comp_results = independent_compressibility_experiment(model, val_loader, device)

    # Assemble results
    results = {
        "mutual_information_bits": mi,
        "pca_semantic_95dim": pca_sem_dim,
        "pca_phase_95dim": pca_phase_dim,
        "cross_predict_phase_to_semantic": cp,
        "cross_predict_semantic_to_phase": cp_rev,
        "resonance_entropy": r_entropy,
        "effective_rank": eff_r,
        "alpha_mean": float(streams["alpha"].mean()),
        "alpha_std": float(streams["alpha"].std()),
        "independent_compression": comp_results,
    }

    # Save JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Gestalt] Results saved to {args.output_json}")

    # Plot
    plot_gestalt_analysis(streams, model, args.output_plot, dataset=dataset)
    print("[Gestalt] Done.")


if __name__ == "__main__":
    main()
