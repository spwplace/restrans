"""
run_gestalt.py
==============
Experiment 4: Gestalt Stream Analysis for trained Medium Resonance model.
"""
import os
import sys
import json
import math
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, '/mnt/agents/output/resonance')
from models import ResonanceTransformer
from utils.data import SyntheticTokenDataset, get_dataloader
from utils.training import compute_perplexity


class StructuredSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size=5000, seq_len=64, num_samples=600, num_modes=10, seed=None):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_modes = num_modes
        self.tokens_per_mode = vocab_size // num_modes
        if seed is not None:
            import random
            random.seed(seed)
            torch.manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        import random
        rng = random.Random(idx + 42)
        x = torch.zeros(self.seq_len, dtype=torch.long)
        mode = rng.randint(0, self.num_modes - 1)
        x[0] = mode * self.tokens_per_mode + rng.randint(0, self.tokens_per_mode - 1)
        for t in range(1, self.seq_len):
            if t % 8 == 0 and t >= 8:
                prev_mode = min((x[t-8].item() // self.tokens_per_mode), self.num_modes - 1)
                mode = prev_mode if rng.random() < 0.7 else rng.randint(0, self.num_modes - 1)
            else:
                mode = mode if rng.random() < 0.7 else rng.randint(0, self.num_modes - 1)
            base = mode * self.tokens_per_mode
            if rng.random() < 0.8:
                prev_in_mode = x[t-1].item() - base
                token = base + ((prev_in_mode + rng.randint(-3, 3)) % self.tokens_per_mode)
            else:
                token = base + rng.randint(0, self.tokens_per_mode - 1)
            x[t] = max(0, min(token, self.vocab_size - 1))
        y = torch.roll(x, shifts=-1, dims=0)
        y[-1] = rng.randint(0, self.vocab_size - 1)
        return x, y


def quantize_tensor(w, bits):
    w_min, w_max = w.min(), w.max()
    scale = (w_max - w_min) / (2**bits - 1) if w_max > w_min else 1.0
    w_quant = ((w - w_min) / scale).round().clamp(0, 2**bits - 1)
    return w_quant * scale + w_min


def effective_rank(matrix):
    """Effective rank = exp(entropy of singular value distribution)."""
    # Use numpy SVD
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    s_norm = s / s.sum()
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    return float(np.exp(entropy))


def compute_pca_dim(embeddings, threshold=0.95):
    """Return number of components needed for threshold fraction of variance."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    pca = PCA()
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim = int(np.searchsorted(cumvar, threshold) + 1)
    return dim


def run_experiment4(device="cpu"):
    print("=" * 60)
    print("Experiment 4: Gestalt Stream Analysis")
    print("=" * 60)

    vocab_size = 5000
    seq_len = 64
    val_ds = StructuredSyntheticDataset(vocab_size, seq_len, num_samples=600, seed=123)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)

    model = ResonanceTransformer(
        vocab_size=vocab_size, d_model=256, nhead=4, num_layers=4,
        dim_feedforward=1024, max_seq_len=seq_len,
    )
    ckpt_path = "/mnt/agents/output/resonance/checkpoints/exp1_medium_resonance.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found. Cannot run Experiment 4.")
        return {}

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    results = {}

    # 1. Extract semantic and phase streams from a validation batch
    print("\n--- Extracting streams ---")
    with torch.no_grad():
        x, y = next(iter(val_loader))
        x = x.to(device)

        h_sem = model.semantic_embedding(x)       # [B, S, D]
        h_phase_raw = model.phase_embedding(x)    # [B, S, F]
        h_phase = model.phase_proj(h_phase_raw)   # [B, S, D]

        # Flatten over batch and sequence
        sem_flat = h_sem.view(-1, h_sem.size(-1)).cpu().numpy()
        phase_flat = h_phase.view(-1, h_phase.size(-1)).cpu().numpy()
        phase_raw_flat = h_phase_raw.view(-1, h_phase_raw.size(-1)).cpu().numpy()

    results["stream_shapes"] = {
        "semantic": list(h_sem.shape),
        "phase_raw": list(h_phase_raw.shape),
        "phase_proj": list(h_phase.shape),
    }

    # 2. PCA dimensionality
    print("--- PCA dimensionality ---")
    sem_pca_dim = compute_pca_dim(sem_flat, 0.95)
    phase_pca_dim = compute_pca_dim(phase_flat, 0.95)
    phase_raw_pca_dim = compute_pca_dim(phase_raw_flat, 0.95)
    results["pca_dim_95"] = {
        "semantic": int(sem_pca_dim),
        "phase_proj": int(phase_pca_dim),
        "phase_raw": int(phase_raw_pca_dim),
    }
    print(f"  Semantic 95% PCA dim: {sem_pca_dim}")
    print(f"  Phase (proj) 95% PCA dim: {phase_pca_dim}")
    print(f"  Phase (raw) 95% PCA dim: {phase_raw_pca_dim}")

    # 3. Cross-predictability (R² of ridge regression)
    print("--- Cross-predictability ---")
    # Sample for speed
    n_sample = min(5000, sem_flat.shape[0])
    idx = np.random.choice(sem_flat.shape[0], n_sample, replace=False)
    sem_sample = sem_flat[idx]
    phase_sample = phase_flat[idx]

    ridge = Ridge(alpha=1.0)
    ridge.fit(phase_sample, sem_sample)
    sem_pred = ridge.predict(phase_sample)
    r2_phase_to_sem = r2_score(sem_sample, sem_pred)

    ridge2 = Ridge(alpha=1.0)
    ridge2.fit(sem_sample, phase_sample)
    phase_pred = ridge2.predict(sem_sample)
    r2_sem_to_phase = r2_score(phase_sample, phase_pred)

    results["cross_predictability_r2"] = {
        "phase_to_semantic": float(r2_phase_to_sem),
        "semantic_to_phase": float(r2_sem_to_phase),
    }
    print(f"  Phase -> Semantic R²: {r2_phase_to_sem:.4f}")
    print(f"  Semantic -> Phase R²: {r2_sem_to_phase:.4f}")

    # 4. Resonance matrix entropy and effective rank
    print("--- Resonance matrix analysis ---")
    with torch.no_grad():
        # Compute resonance matrix for all tokens in vocab
        all_tokens = torch.arange(vocab_size, device=device)
        phase_emb = model.phase_embedding(all_tokens)  # [V, F]
        p_i = phase_emb.unsqueeze(1)  # [V, 1, F]
        p_j = phase_emb.unsqueeze(0)  # [1, V, F]
        R_full = torch.cos(p_i - p_j).mean(dim=-1).cpu().numpy()  # [V, V]

    # Entropy of resonance matrix (treat as distribution after softmax)
    R_softmax = np.exp(R_full - R_full.max(axis=1, keepdims=True))
    R_softmax /= R_softmax.sum(axis=1, keepdims=True)
    entropy = -np.sum(R_softmax * np.log(R_softmax + 1e-12), axis=1).mean()

    eff_rank = effective_rank(R_full)
    spectral_norm = float(np.linalg.norm(R_full, ord=2))
    frobenius_norm = float(np.linalg.norm(R_full, ord='fro'))

    results["resonance_matrix"] = {
        "entropy": float(entropy),
        "effective_rank": float(eff_rank),
        "spectral_norm": spectral_norm,
        "frobenius_norm": frobenius_norm,
        "shape": list(R_full.shape),
    }
    print(f"  Entropy: {entropy:.4f}")
    print(f"  Effective rank: {eff_rank:.2f}")
    print(f"  Spectral norm: {spectral_norm:.4f}")

    # 5. Independent compressibility
    print("--- Independent compressibility ---")
    orig_state = model.state_dict()

    # Phase INT4, semantic FP32
    state_phase_q = {k: v.clone() for k, v in orig_state.items()}
    for k in state_phase_q:
        if "phase" in k:
            state_phase_q[k] = quantize_tensor(state_phase_q[k], 4)
    model.load_state_dict(state_phase_q)
    ppl_phase_int4 = compute_perplexity(model, val_loader, device=device, max_batches=None)
    print(f"  Phase INT4 (semantic FP32): PPL={ppl_phase_int4:.2f}")

    # Semantic INT4, phase FP32
    state_sem_q = {k: v.clone() for k, v in orig_state.items()}
    for k in state_sem_q:
        if "semantic" in k or ("lm_head" in k and "weight" in k):
            state_sem_q[k] = quantize_tensor(state_sem_q[k], 4)
    model.load_state_dict(state_sem_q)
    ppl_sem_int4 = compute_perplexity(model, val_loader, device=device, max_batches=None)
    print(f"  Semantic INT4 (phase FP32): PPL={ppl_sem_int4:.2f}")

    # Baseline
    model.load_state_dict(orig_state)
    ppl_baseline = compute_perplexity(model, val_loader, device=device, max_batches=None)
    print(f"  Baseline FP32: PPL={ppl_baseline:.2f}")

    results["independent_compressibility"] = {
        "baseline_ppl": ppl_baseline,
        "phase_int4_ppl": ppl_phase_int4,
        "semantic_int4_ppl": ppl_sem_int4,
        "phase_delta": ppl_phase_int4 - ppl_baseline,
        "semantic_delta": ppl_sem_int4 - ppl_baseline,
    }

    # 6. Alpha blend statistics
    alpha = torch.sigmoid(model.alpha).detach().cpu().numpy()
    results["alpha_stats"] = {
        "mean": float(alpha.mean()),
        "std": float(alpha.std()),
        "min": float(alpha.min()),
        "max": float(alpha.max()),
    }
    print(f"  Alpha mean={alpha.mean():.4f} std={alpha.std():.4f}")

    out_path = "/mnt/agents/output/resonance/results/exp4_gestalt.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment 4 complete. Results saved to {out_path}")
    return results


if __name__ == "__main__":
    torch.set_num_threads(4)
    run_experiment4(device="cpu")
