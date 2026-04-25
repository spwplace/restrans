"""
Comprehensive analysis script for Resonance Transformer experiments.
Tasks:
  1. Generate scaling plot and training curves
  2. Compressibility analysis on medium models
  3. Gestalt analysis on medium resonance model
"""
import os
import sys
import json
import math
import time
import random
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/agents/output/resonance')
from models import StandardTransformer, ResonanceTransformer

OUTPUT_DIR = "/mnt/agents/output/resonance"
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticTokenDataset(Dataset):
    """Structured synthetic tokens with modes and long-range dependencies."""
    def __init__(self, vocab_size=5000, seq_len=64, num_samples=2000, num_modes=10, seed=None):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_modes = num_modes
        self.tokens_per_mode = vocab_size // num_modes
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def _mode_of(self, token):
        return min(token // self.tokens_per_mode, self.num_modes - 1)

    def __getitem__(self, idx):
        rng = random.Random(idx + 42)
        x = torch.zeros(self.seq_len, dtype=torch.long)
        mode = rng.randint(0, self.num_modes - 1)
        x[0] = mode * self.tokens_per_mode + rng.randint(0, self.tokens_per_mode - 1)
        for t in range(1, self.seq_len):
            if t % 8 == 0 and t >= 8:
                prev_mode = self._mode_of(x[t-8].item())
                mode = prev_mode if rng.random() < 0.7 else rng.randint(0, self.num_modes - 1)
            else:
                mode = mode if rng.random() < 0.7 else rng.randint(0, self.num_modes - 1)
            base = mode * self.tokens_per_mode
            if rng.random() < 0.8:
                prev_in_mode = x[t-1].item() - base
                delta = rng.randint(-3, 3)
                token = base + ((prev_in_mode + delta) % self.tokens_per_mode)
            else:
                token = base + rng.randint(0, self.tokens_per_mode - 1)
            x[t] = max(0, min(token, self.vocab_size - 1))
        y = torch.roll(x, shifts=-1, dims=0)
        y[-1] = rng.randint(0, self.vocab_size - 1)
        return x, y


def make_loader(num_samples=2000, batch_size=32, seed=None):
    ds = SyntheticTokenDataset(num_samples=num_samples, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def compute_ppl(model, dataloader, device='cpu', max_batches=10):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
            count += 1
            if max_batches is not None and count >= max_batches:
                break
    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)


# ---------------------------------------------------------------------------
# TASK 1: Plots
# ---------------------------------------------------------------------------

def task1_scaling_plot():
    print("\n" + "="*60)
    print("TASK 1: Generating plots")
    print("="*60)

    # --- Scaling law plot ---
    exp2_files = {
        '1x_standard': 'results/exp2_1x_standard.json',
        '1x_resonance': 'results/exp2_1x_resonance.json',
        '2x_standard': 'results/exp2_2x_standard.json',
        '2x_resonance': 'results/exp2_2x_resonance.json',
        '4x_standard': 'results/exp2_4x_standard.json',
        '4x_resonance': 'results/exp2_4x_resonance.json',
    }

    standard_pts = []
    resonance_pts = []

    for key, path in exp2_files.items():
        full_path = os.path.join(OUTPUT_DIR, path)
        if not os.path.exists(full_path):
            print(f"  SKIP missing {path}")
            continue
        with open(full_path) as f:
            data = json.load(f)
        params = data.get('params', data.get('n_params'))
        ppl = data['final_val_ppl']
        if 'standard' in key:
            standard_pts.append((params, ppl))
        else:
            resonance_pts.append((params, ppl))

    # Also try exp2_4x_resonance from checkpoint if json missing
    exp2_4x_res_path = os.path.join(OUTPUT_DIR, 'results/exp2_4x_resonance.json')
    if not os.path.exists(exp2_4x_res_path):
        ckpt_path = os.path.join(OUTPUT_DIR, 'checkpoints/exp2_4x_resonance.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # Need to compute from model
            model = ResonanceTransformer(vocab_size=5000, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024, max_seq_len=64)
            model.load_state_dict(ckpt['model_state_dict'])
            params = sum(p.numel() for p in model.parameters())
            # Evaluate
            val_loader = make_loader(num_samples=200, batch_size=32, seed=123)
            ppl = compute_ppl(model, val_loader, max_batches=10)
            resonance_pts.append((params, ppl))
            print(f"  Computed exp2_4x_resonance from checkpoint: params={params}, ppl={ppl:.2f}")

    fig, ax = plt.subplots(figsize=(8, 6))

    if standard_pts:
        xs, ys = zip(*sorted(standard_pts))
        logxs = [math.log10(x) for x in xs]
        logys = [math.log10(y) for y in ys]
        ax.plot(logxs, logys, 'o-', label='Standard', linewidth=2, markersize=8, color='#1f77b4')
        for i, (lx, ly) in enumerate(zip(logxs, logys)):
            ax.scatter([lx], [ly], s=120, zorder=5, color='#1f77b4', edgecolors='black', linewidth=1.5)
        # Reference line: linear fit
        if len(logxs) >= 2:
            coeffs = np.polyfit(logxs, logys, 1)
            ref_x = np.linspace(min(logxs)-0.1, max(logxs)+0.1, 100)
            ref_y = np.polyval(coeffs, ref_x)
            ax.plot(ref_x, ref_y, '--', color='#1f77b4', alpha=0.5, linewidth=1.5, label='Standard trend')

    if resonance_pts:
        xs, ys = zip(*sorted(resonance_pts))
        logxs = [math.log10(x) for x in xs]
        logys = [math.log10(y) for y in ys]
        ax.plot(logxs, logys, 's--', label='Resonance', linewidth=2, markersize=8, color='#ff7f0e')
        for i, (lx, ly) in enumerate(zip(logxs, logys)):
            ax.scatter([lx], [ly], s=120, zorder=5, color='#ff7f0e', edgecolors='black', linewidth=1.5)
        if len(logxs) >= 2:
            coeffs = np.polyfit(logxs, logys, 1)
            ref_x = np.linspace(min(logxs)-0.1, max(logxs)+0.1, 100)
            ref_y = np.polyval(coeffs, ref_x)
            ax.plot(ref_x, ref_y, '--', color='#ff7f0e', alpha=0.5, linewidth=1.5, label='Resonance trend')

    ax.set_xlabel("log10(Parameter Count)", fontsize=12)
    ax.set_ylabel("log10(Final Validation Perplexity)", fontsize=12)
    ax.set_title("Scaling Laws: Standard vs Resonance Transformers", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "scaling_law.png"), dpi=200)
    plt.close(fig)
    print("  Saved figures/scaling_law.png")

    # --- Training curves plot ---
    exp1_keys = [
        ('small_standard', 'results/exp1_small_standard.json'),
        ('small_resonance', 'results/exp1_small_resonance.json'),
        ('medium_standard', 'results/exp1_medium_standard.json'),
        ('medium_resonance', 'results/exp1_medium_resonance.json'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    configs = [('small', 'Small'), ('medium', 'Medium')]
    for idx, (size, size_label) in enumerate(configs):
        ax_loss = axes[idx, 0]
        ax_ppl = axes[idx, 1]

        for variant in ['standard', 'resonance']:
            key = f"{size}_{variant}"
            json_path = os.path.join(OUTPUT_DIR, f"results/exp1_{key}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)
            hist = data['history']
            epochs = hist.get('epochs', list(range(1, len(hist['train_loss'])+1)))
            style = '-' if variant == 'standard' else '--'
            color = '#1f77b4' if variant == 'standard' else '#ff7f0e'
            label = variant.title()
            ax_loss.plot(epochs, hist['train_loss'], style, label=label, linewidth=2, color=color)
            ax_ppl.plot(epochs, hist['val_ppl'], style, label=label, linewidth=2, color=color)

        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Train Loss")
        ax_loss.set_title(f"{size_label}: Training Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        ax_ppl.set_xlabel("Epoch")
        ax_ppl.set_ylabel("Validation Perplexity")
        ax_ppl.set_title(f"{size_label}: Validation Perplexity")
        ax_ppl.legend()
        ax_ppl.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "training_curves.png"), dpi=200)
    plt.close(fig)
    print("  Saved figures/training_curves.png")


# ---------------------------------------------------------------------------
# TASK 2: Compressibility
# ---------------------------------------------------------------------------

def quantize_uniform(model, n_bits):
    """Simulate uniform quantization by rounding weights to n_bits precision."""
    scale = 2 ** n_bits
    # Quantize all float parameters
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype.is_floating_point:
                p_min, p_max = p.min().item(), p.max().item()
                if p_max > p_min:
                    q = torch.round((p - p_min) / (p_max - p_min) * (scale - 1))
                    p.copy_(q / (scale - 1) * (p_max - p_min) + p_min)


def prune_magnitude(model, sparsity):
    """Magnitude pruning: zero out smallest magnitude weights."""
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() == 0:
                continue
            flat = p.abs().view(-1)
            k = int(sparsity * flat.numel())
            if k > 0:
                threshold = flat.kthvalue(k).values.item()
                mask = flat > threshold
                p.view(-1)[~mask] = 0.0


def get_state_dict_copy(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def restore_state_dict(model, state_dict):
    model.load_state_dict(state_dict)


def task2_compressibility():
    print("\n" + "="*60)
    print("TASK 2: Compressibility on Medium Models")
    print("="*60)

    device = 'cpu'
    val_loader = make_loader(num_samples=2000, batch_size=32, seed=123)

    configs = {
        'standard': {'cls': StandardTransformer, 'path': 'checkpoints/exp1_medium_standard.pt',
                     'kwargs': {'vocab_size': 5000, 'd_model': 256, 'nhead': 4, 'num_layers': 4, 'dim_feedforward': 1024, 'max_seq_len': 64}},
        'resonance': {'cls': ResonanceTransformer, 'path': 'checkpoints/exp1_medium_resonance.pt',
                      'kwargs': {'vocab_size': 5000, 'd_model': 256, 'nhead': 4, 'num_layers': 4, 'dim_feedforward': 1024, 'max_seq_len': 64}},
    }

    results = {}
    total_params_baseline = {}

    for variant, info in configs.items():
        print(f"\n--- {variant.title()} ---")
        model = info['cls'](**info['kwargs'])
        ckpt = torch.load(os.path.join(OUTPUT_DIR, info['path']), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()

        baseline_state = get_state_dict_copy(model)
        n_params = sum(p.numel() for p in model.parameters())
        total_params_baseline[variant] = n_params

        # Baseline PPL
        baseline_ppl = compute_ppl(model, val_loader, device, max_batches=10)
        print(f"  Baseline FP32 PPL: {baseline_ppl:.2f} (params={n_params:,})")

        tests = []

        # Uniform quantization
        for bits in [8, 4, 2]:
            restore_state_dict(model, baseline_state)
            quantize_uniform(model, bits)
            ppl = compute_ppl(model, val_loader, device, max_batches=10)
            ratio = 32 / bits
            tests.append({
                'method': f'uniform_{bits}bit',
                'compression_ratio': ratio,
                'bits': bits,
                'ppl': ppl,
                'delta_ppl': ppl - baseline_ppl,
            })
            print(f"  Uniform {bits}-bit: PPL={ppl:.2f}, ratio={ratio:.1f}x")

        # Magnitude pruning
        for sparsity in [0.3, 0.5, 0.7]:
            restore_state_dict(model, baseline_state)
            prune_magnitude(model, sparsity)
            ppl = compute_ppl(model, val_loader, device, max_batches=10)
            # Compression ratio: sparse storage ~ 1/(1-sparsity) with CSR, but approximate
            ratio = 1.0 / (1.0 - sparsity)
            tests.append({
                'method': f'prune_{int(sparsity*100)}pct',
                'compression_ratio': ratio,
                'sparsity': sparsity,
                'ppl': ppl,
                'delta_ppl': ppl - baseline_ppl,
            })
            print(f"  Prune {int(sparsity*100)}%: PPL={ppl:.2f}, ratio~={ratio:.1f}x")

        if variant == 'resonance':
            # Phase-only aggressive quantization
            restore_state_dict(model, baseline_state)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if 'phase_embedding' in name or 'phase_proj' in name:
                        p_min, p_max = p.min().item(), p.max().item()
                        scale = 2 ** 2  # 2-bit
                        if p_max > p_min:
                            q = torch.round((p - p_min) / (p_max - p_min) * (scale - 1))
                            p.copy_(q / (scale - 1) * (p_max - p_min) + p_min)
                    elif 'semantic_embedding' in name:
                        p_min, p_max = p.min().item(), p.max().item()
                        scale = 2 ** 8  # 8-bit
                        if p_max > p_min:
                            q = torch.round((p - p_min) / (p_max - p_min) * (scale - 1))
                            p.copy_(q / (scale - 1) * (p_max - p_min) + p_min)
            ppl = compute_ppl(model, val_loader, device, max_batches=10)
            # Approximate compression: phase 2-bit, semantic 8-bit, rest 32-bit
            tests.append({
                'method': 'phase_2bit_semantic_8bit',
                'compression_ratio': 3.5,  # rough estimate
                'ppl': ppl,
                'delta_ppl': ppl - baseline_ppl,
            })
            print(f"  Phase 2-bit + Semantic 8-bit: PPL={ppl:.2f}")

            # Vice versa: semantic 2-bit, phase 8-bit
            restore_state_dict(model, baseline_state)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if 'semantic_embedding' in name:
                        p_min, p_max = p.min().item(), p.max().item()
                        scale = 2 ** 2
                        if p_max > p_min:
                            q = torch.round((p - p_min) / (p_max - p_min) * (scale - 1))
                            p.copy_(q / (scale - 1) * (p_max - p_min) + p_min)
                    elif 'phase_embedding' in name or 'phase_proj' in name:
                        p_min, p_max = p.min().item(), p.max().item()
                        scale = 2 ** 8
                        if p_max > p_min:
                            q = torch.round((p - p_min) / (p_max - p_min) * (scale - 1))
                            p.copy_(q / (scale - 1) * (p_max - p_min) + p_min)
            ppl = compute_ppl(model, val_loader, device, max_batches=10)
            tests.append({
                'method': 'semantic_2bit_phase_8bit',
                'compression_ratio': 3.5,
                'ppl': ppl,
                'delta_ppl': ppl - baseline_ppl,
            })
            print(f"  Semantic 2-bit + Phase 8-bit: PPL={ppl:.2f}")

        results[variant] = {
            'baseline_ppl': baseline_ppl,
            'params': n_params,
            'compression_tests': tests,
        }

    # Save results
    with open(os.path.join(OUTPUT_DIR, "results", "compressibility_real.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved results/compressibility_real.json")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'standard': '#1f77b4', 'resonance': '#ff7f0e'}
    markers = {'standard': 'o', 'resonance': 's'}

    for variant in ['standard', 'resonance']:
        if variant not in results:
            continue
        tests = results[variant]['compression_tests']
        crs = [t['compression_ratio'] for t in tests]
        ppls = [t['ppl'] for t in tests]
        labels = [t['method'] for t in tests]
        ax.scatter(crs, ppls, label=variant.title(), color=colors[variant],
                   marker=markers[variant], s=100, alpha=0.8, edgecolors='black', linewidth=1)
        for cr, ppl, lab in zip(crs, ppls, labels):
            ax.annotate(lab, (cr, ppl), fontsize=7, alpha=0.8,
                        textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Compression Ratio (higher = more compressed)", fontsize=12)
    ax.set_ylabel("Validation Perplexity", fontsize=12)
    ax.set_title("Compressibility: Medium Models", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "compressibility_real.png"), dpi=200)
    plt.close(fig)
    print("  Saved figures/compressibility_real.png")

    return results


# ---------------------------------------------------------------------------
# TASK 3: Gestalt Analysis
# ---------------------------------------------------------------------------

def task3_gestalt():
    print("\n" + "="*60)
    print("TASK 3: Gestalt Analysis on Medium Resonance")
    print("="*60)

    device = 'cpu'
    model = ResonanceTransformer(
        vocab_size=5000, d_model=256, nhead=4, num_layers=4,
        dim_feedforward=1024, max_seq_len=64, n_frequencies=32
    )
    ckpt = torch.load(os.path.join(OUTPUT_DIR, 'checkpoints/exp1_medium_resonance.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Extract from batch of 128 tokens
    batch_size = 128
    seq_len = 64
    tokens = torch.randint(0, 5000, (batch_size, seq_len), device=device)

    with torch.no_grad():
        h_sem = model.semantic_embedding(tokens).cpu().numpy()          # [B, S, D]
        h_phase_raw = model.phase_embedding(tokens).cpu().numpy()         # [B, S, F]
        h_phase = model.phase_proj(torch.tensor(h_phase_raw)).cpu().numpy()  # [B, S, D]
        R = model.compute_resonance_matrix(tokens).cpu().numpy()          # [B, S, S]
        alpha = torch.sigmoid(model.alpha).cpu().numpy()                  # [D]

    # Flatten over batch and seq for PCA
    h_sem_flat = h_sem.reshape(-1, h_sem.shape[-1])      # [B*S, D]
    h_phase_flat = h_phase.reshape(-1, h_phase.shape[-1])  # [B*S, D]
    h_phase_raw_flat = h_phase_raw.reshape(-1, h_phase_raw.shape[-1])  # [B*S, F]

    print(f"  h_sem shape: {h_sem.shape}, h_phase shape: {h_phase.shape}, R shape: {R.shape}")

    # --- PCA 95% variance ---
    def pca_95_dims(X):
        pca = PCA().fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dims = np.argmax(cumvar >= 0.95) + 1
        return dims, pca

    dims_sem, pca_sem = pca_95_dims(h_sem_flat)
    dims_phase, pca_phase = pca_95_dims(h_phase_flat)
    dims_phase_raw, pca_phase_raw = pca_95_dims(h_phase_raw_flat)
    print(f"  PCA 95% dims: semantic={dims_sem}, phase_proj={dims_phase}, phase_raw={dims_phase_raw}")

    # --- Cross-predictability R² ---
    def ridge_r2(X, Y):
        """Ridge regression R² predicting Y from X."""
        from sklearn.model_selection import cross_val_score
        clf = Ridge(alpha=1.0)
        # Use cross-validated R²
        scores = cross_val_score(clf, X, Y, cv=3, scoring='r2')
        return float(np.mean(scores))

    # Sample for speed
    sample_idx = np.random.choice(len(h_sem_flat), size=min(2000, len(h_sem_flat)), replace=False)
    X_sem = h_sem_flat[sample_idx]
    X_phase = h_phase_flat[sample_idx]

    r2_phase_to_sem = ridge_r2(X_phase, X_sem)
    r2_sem_to_phase = ridge_r2(X_sem, X_phase)
    print(f"  Cross-predictability R²: phase->semantic={r2_phase_to_sem:.4f}, semantic->phase={r2_sem_to_phase:.4f}")

    # --- Resonance matrix entropy ---
    # Use first batch item
    R0 = R[0]  # [S, S]
    # Entropy per row (softmaxed)
    R_softmax = torch.softmax(torch.tensor(R0), dim=-1).numpy()
    row_entropies = -np.sum(R_softmax * np.log(R_softmax + 1e-12), axis=-1)
    mean_entropy = float(np.mean(row_entropies))
    print(f"  Resonance matrix mean entropy: {mean_entropy:.4f}")

    # --- Effective rank of R ---
    U, s, Vt = np.linalg.svd(R0)
    p = s / s.sum()
    effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    print(f"  Effective rank of R: {effective_rank:.2f}")

    # --- Independent compressibility ---
    val_loader = make_loader(num_samples=2000, batch_size=32, seed=123)
    baseline_state = get_state_dict_copy(model)

    # Baseline
    baseline_ppl = compute_ppl(model, val_loader, device, max_batches=10)
    print(f"  Baseline FP32 PPL: {baseline_ppl:.2f}")

    # Quantize phase embeddings to 4 buckets, semantic FP32
    restore_state_dict(model, baseline_state)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'phase_embedding' in name or 'phase_proj' in name:
                p_min, p_max = p.min().item(), p.max().item()
                scale = 4  # 4 buckets = 2-bit
                if p_max > p_min:
                    q = torch.round((p - p_min) / (p_max - p_min) * (scale - 1))
                    p.copy_(q / (scale - 1) * (p_max - p_min) + p_min)
    ppl_phase_q = compute_ppl(model, val_loader, device, max_batches=10)
    print(f"  Phase quantized (4 buckets): PPL={ppl_phase_q:.2f}")

    # Quantize semantic embeddings to 4 buckets, phase FP32
    restore_state_dict(model, baseline_state)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'semantic_embedding' in name:
                p_min, p_max = p.min().item(), p.max().item()
                scale = 4
                if p_max > p_min:
                    q = torch.round((p - p_min) / (p_max - p_min) * (scale - 1))
                    p.copy_(q / (scale - 1) * (p_max - p_min) + p_min)
    ppl_sem_q = compute_ppl(model, val_loader, device, max_batches=10)
    print(f"  Semantic quantized (4 buckets): PPL={ppl_sem_q:.2f}")

    # --- Save results ---
    results = {
        'pca_dim_95': {
            'semantic': int(dims_sem),
            'phase_proj': int(dims_phase),
            'phase_raw': int(dims_phase_raw),
        },
        'pca_explained_variance': {
            'semantic': pca_sem.explained_variance_ratio_.tolist(),
            'phase_proj': pca_phase.explained_variance_ratio_.tolist(),
            'phase_raw': pca_phase_raw.explained_variance_ratio_.tolist(),
        },
        'cross_predictability_r2': {
            'phase_to_semantic': r2_phase_to_sem,
            'semantic_to_phase': r2_sem_to_phase,
        },
        'resonance_matrix': {
            'entropy': mean_entropy,
            'effective_rank': effective_rank,
            'shape': list(R.shape),
        },
        'alpha_stats': {
            'mean': float(alpha.mean()),
            'std': float(alpha.std()),
            'min': float(alpha.min()),
            'max': float(alpha.max()),
            'values': alpha.tolist(),
        },
        'independent_compressibility': {
            'baseline_ppl': baseline_ppl,
            'phase_quantized_ppl': ppl_phase_q,
            'semantic_quantized_ppl': ppl_sem_q,
            'phase_delta': ppl_phase_q - baseline_ppl,
            'semantic_delta': ppl_sem_q - baseline_ppl,
        },
    }

    with open(os.path.join(OUTPUT_DIR, "results", "gestalt_real.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved results/gestalt_real.json")

    # --- Multi-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: PCA scree plot for semantic
    ax = axes[0, 0]
    cumvar_sem = np.cumsum(pca_sem.explained_variance_ratio_)
    ax.plot(range(1, len(cumvar_sem)+1), cumvar_sem, '-', color='#1f77b4', linewidth=2)
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95% variance')
    ax.axvline(dims_sem, color='green', linestyle='--', alpha=0.5, label=f'{dims_sem} dims')
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Panel 1: Semantic Stream PCA")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: PCA scree plot for phase
    ax = axes[0, 1]
    cumvar_phase = np.cumsum(pca_phase.explained_variance_ratio_)
    ax.plot(range(1, len(cumvar_phase)+1), cumvar_phase, '-', color='#ff7f0e', linewidth=2)
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95% variance')
    ax.axvline(dims_phase, color='green', linestyle='--', alpha=0.5, label=f'{dims_phase} dims')
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Panel 2: Phase Stream PCA")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Alpha distribution histogram
    ax = axes[1, 0]
    ax.hist(alpha, bins=30, color='purple', edgecolor='black', alpha=0.7)
    ax.axvline(alpha.mean(), color='red', linestyle='--', linewidth=2, label=f'mean={alpha.mean():.3f}')
    ax.set_xlabel("Alpha (blend coefficient)")
    ax.set_ylabel("Count")
    ax.set_title("Panel 3: Alpha Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Resonance matrix heatmap (first 16x16)
    ax = axes[1, 1]
    subR = R0[:16, :16]
    im = ax.imshow(subR, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xlabel("Token position j")
    ax.set_ylabel("Token position i")
    ax.set_title("Panel 4: Resonance Matrix (16x16)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "gestalt_real.png"), dpi=200)
    plt.close(fig)
    print("  Saved figures/gestalt_real.png")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_num_threads(4)
    np.random.seed(42)
    torch.manual_seed(42)

    task1_scaling_plot()
    compress_results = task2_compressibility()
    gestalt_results = task3_gestalt()

    print("\n" + "="*60)
    print("ALL TASKS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  figures/scaling_law.png")
    print("  figures/training_curves.png")
    print("  results/compressibility_real.json")
    print("  figures/compressibility_real.png")
    print("  results/gestalt_real.json")
    print("  figures/gestalt_real.png")

    # Print numeric summaries
    print("\n--- TASK 2: Compressibility Results ---")
    for variant, data in compress_results.items():
        print(f"\n{variant.title()}:")
        print(f"  Baseline PPL: {data['baseline_ppl']:.2f}")
        for t in data['compression_tests']:
            print(f"  {t['method']}: PPL={t['ppl']:.2f}, ratio={t['compression_ratio']:.2f}x")

    print("\n--- TASK 3: Gestalt Results ---")
    g = gestalt_results
    print(f"  PCA 95% dims: semantic={g['pca_dim_95']['semantic']}, phase_proj={g['pca_dim_95']['phase_proj']}, phase_raw={g['pca_dim_95']['phase_raw']}")
    print(f"  Cross-predictability R²: phase→sem={g['cross_predictability_r2']['phase_to_semantic']:.4f}, sem→phase={g['cross_predictability_r2']['semantic_to_phase']:.4f}")
    print(f"  Resonance entropy: {g['resonance_matrix']['entropy']:.4f}")
    print(f"  Effective rank: {g['resonance_matrix']['effective_rank']:.2f}")
    print(f"  Alpha mean={g['alpha_stats']['mean']:.4f}, std={g['alpha_stats']['std']:.4f}")
    print(f"  Independent compressibility:")
    print(f"    Baseline FP32: {g['independent_compressibility']['baseline_ppl']:.2f}")
    print(f"    Phase quantized: {g['independent_compressibility']['phase_quantized_ppl']:.2f} (Δ={g['independent_compressibility']['phase_delta']:.4f})")
    print(f"    Semantic quantized: {g['independent_compressibility']['semantic_quantized_ppl']:.2f} (Δ={g['independent_compressibility']['semantic_delta']:.4f})")
