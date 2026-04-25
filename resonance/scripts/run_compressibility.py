"""
run_compressibility.py
=======================
Experiment 3: Compressibility Test for trained Medium models.
"""
import os
import sys
import json
import math
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

sys.path.insert(0, '/mnt/agents/output/resonance')
from models import StandardTransformer, ResonanceTransformer
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
    """Simulate quantization by bucketing."""
    w_min, w_max = w.min(), w.max()
    scale = (w_max - w_min) / (2**bits - 1) if w_max > w_min else 1.0
    w_quant = ((w - w_min) / scale).round().clamp(0, 2**bits - 1)
    w_dequant = w_quant * scale + w_min
    return w_dequant


def quantize_model_state_dict(state_dict, bits, exclude_patterns=None):
    """Quantize all float tensors in state dict."""
    if exclude_patterns is None:
        exclude_patterns = []
    quantized = OrderedDict()
    for key, val in state_dict.items():
        if val.dtype == torch.float32 and not any(p in key for p in exclude_patterns):
            quantized[key] = quantize_tensor(val, bits)
        else:
            quantized[key] = val.clone()
    return quantized


def prune_model_state_dict(state_dict, sparsity):
    """Magnitude pruning: zero out smallest magnitude weights."""
    pruned = OrderedDict()
    for key, val in state_dict.items():
        if val.dtype == torch.float32 and val.dim() >= 2:  # only prune matrices
            flat = val.abs().view(-1)
            k = int(sparsity * flat.numel())
            if k > 0:
                threshold = torch.kthvalue(flat, k)[0].item()
                mask = val.abs() >= threshold
                pruned[key] = val * mask
            else:
                pruned[key] = val.clone()
        else:
            pruned[key] = val.clone()
    return pruned


def evaluate_quantized(model_class, state_dict, config, val_loader, device):
    model = model_class(**config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return compute_perplexity(model, val_loader, device=device, max_batches=None)


def run_experiment3(device="cpu"):
    print("=" * 60)
    print("Experiment 3: Compressibility Test")
    print("=" * 60)

    vocab_size = 5000
    seq_len = 64
    val_ds = StructuredSyntheticDataset(vocab_size, seq_len, num_samples=600, seed=123)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)

    base_config = {
        "vocab_size": vocab_size,
        "d_model": 256,
        "nhead": 4,
        "num_layers": 4,
        "dim_feedforward": 1024,
        "max_seq_len": seq_len,
    }

    results = {}

    for variant in ["standard", "resonance"]:
        model_name = f"exp1_medium_{variant}"
        ckpt_path = f"/mnt/agents/output/resonance/checkpoints/{model_name}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found, skipping {variant}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        orig_state = ckpt["model_state_dict"]

        if variant == "standard":
            model_class = StandardTransformer
        else:
            model_class = ResonanceTransformer

        # Baseline (FP32)
        ppl_fp32 = evaluate_quantized(model_class, orig_state, base_config, val_loader, device)
        print(f"\n{variant} FP32 baseline PPL: {ppl_fp32:.2f}")
        results[variant] = {"baseline_ppl": ppl_fp32, "compression_tests": []}

        # INT8 quantization
        state_int8 = quantize_model_state_dict(orig_state, bits=8)
        ppl_int8 = evaluate_quantized(model_class, state_int8, base_config, val_loader, device)
        results[variant]["compression_tests"].append({
            "method": "INT8", "compression_ratio": 4.0, "ppl": ppl_int8,
            "params_affected": "all",
        })
        print(f"  INT8 PPL: {ppl_int8:.2f}")

        # INT4 quantization
        state_int4 = quantize_model_state_dict(orig_state, bits=4)
        ppl_int4 = evaluate_quantized(model_class, state_int4, base_config, val_loader, device)
        results[variant]["compression_tests"].append({
            "method": "INT4", "compression_ratio": 8.0, "ppl": ppl_int4,
            "params_affected": "all",
        })
        print(f"  INT4 PPL: {ppl_int4:.2f}")

        # Magnitude pruning
        for sparsity in [0.3, 0.5, 0.7]:
            state_pruned = prune_model_state_dict(orig_state, sparsity)
            ppl_pruned = evaluate_quantized(model_class, state_pruned, base_config, val_loader, device)
            cr = 1.0 / (1.0 - sparsity)
            results[variant]["compression_tests"].append({
                "method": f"prune_{sparsity}", "compression_ratio": cr, "ppl": ppl_pruned,
                "params_affected": "all",
            })
            print(f"  Prune {sparsity} PPL: {ppl_pruned:.2f}")

        # Resonance-specific: phase-only and semantic-only quantization
        if variant == "resonance":
            # INT4 on phase embeddings only, semantic stays FP32
            state_phase_int4 = quantize_model_state_dict(orig_state, bits=4,
                                                          exclude_patterns=["semantic", "lm_head", "pos_encoder"])
            ppl_phase_int4 = evaluate_quantized(model_class, state_phase_int4, base_config, val_loader, device)
            # Rough estimate: phase embeddings are ~1.6% of params
            total_params = sum(v.numel() for v in orig_state.values())
            phase_params = sum(v.numel() for k, v in orig_state.items() if "phase" in k)
            cr_phase = total_params / (total_params - phase_params * 0.875)
            results[variant]["compression_tests"].append({
                "method": "phase_only_INT4", "compression_ratio": cr_phase, "ppl": ppl_phase_int4,
                "params_affected": "phase_only",
            })
            print(f"  Phase-only INT4 PPL: {ppl_phase_int4:.2f}")

            # INT2 on phase embeddings only
            state_phase_int2 = quantize_model_state_dict(orig_state, bits=2,
                                                          exclude_patterns=["semantic", "lm_head", "pos_encoder"])
            ppl_phase_int2 = evaluate_quantized(model_class, state_phase_int2, base_config, val_loader, device)
            cr_phase2 = total_params / (total_params - phase_params * 0.9375)
            results[variant]["compression_tests"].append({
                "method": "phase_only_INT2", "compression_ratio": cr_phase2, "ppl": ppl_phase_int2,
                "params_affected": "phase_only",
            })
            print(f"  Phase-only INT2 PPL: {ppl_phase_int2:.2f}")

            # INT4 on semantic embeddings only
            state_sem_int4 = quantize_model_state_dict(orig_state, bits=4,
                                                        exclude_patterns=["phase", "lm_head", "pos_encoder"])
            ppl_sem_int4 = evaluate_quantized(model_class, state_sem_int4, base_config, val_loader, device)
            sem_params = sum(v.numel() for k, v in orig_state.items() if "semantic" in k)
            cr_sem = total_params / (total_params - sem_params * 0.875)
            results[variant]["compression_tests"].append({
                "method": "semantic_only_INT4", "compression_ratio": cr_sem, "ppl": ppl_sem_int4,
                "params_affected": "semantic_only",
            })
            print(f"  Semantic-only INT4 PPL: {ppl_sem_int4:.2f}")

    out_path = "/mnt/agents/output/resonance/results/exp3_compressibility.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment 3 complete. Results saved to {out_path}")
    return results


if __name__ == "__main__":
    torch.set_num_threads(4)
    run_experiment3(device="cpu")
