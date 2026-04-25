"""
run_real_experiments.py
=======================
Experiment 1: Real Data Comparison (Primary)
Train both Standard and Resonance models on structured synthetic data.

NOTE: For CPU compute constraints, we use:
  - seq_len = 64 (spec says 128, but 64 is ~2x faster on CPU)
  - train_samples = 3000, val_samples = 600 (reduced from spec for speed)
  - epochs calibrated per model size to fit within ~10 min per run
"""
import os
import sys
import time
import json
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/mnt/agents/output/resonance')
from models import StandardTransformer, ResonanceTransformer
from utils.training import train_model, compute_perplexity, compute_accuracy_at_k

# ---------------------------------------------------------------------------
# Richer synthetic dataset with modes and long-range dependencies
# ---------------------------------------------------------------------------

class StructuredSyntheticDataset(Dataset):
    """
    Synthetic token sequences with:
      - Multiple modes (clusters of co-occurring tokens)
      - Local Markov structure within modes
      - Long-range dependencies (every 8th token correlates)
    """
    def __init__(self, vocab_size=5000, seq_len=64, num_samples=3000, num_modes=10, seed=None):
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
        # Seed per sample for determinism
        rng = random.Random(idx + 42)
        torch_rng = torch.Generator().manual_seed(idx + 42)

        x = torch.zeros(self.seq_len, dtype=torch.long)
        # Start with random mode
        mode = rng.randint(0, self.num_modes - 1)
        x[0] = mode * self.tokens_per_mode + rng.randint(0, self.tokens_per_mode - 1)

        for t in range(1, self.seq_len):
            # Every 8th token correlates with token 8 positions back
            if t % 8 == 0 and t >= 8:
                prev_mode = self._mode_of(x[t-8].item())
                if rng.random() < 0.7:
                    mode = prev_mode
                else:
                    mode = rng.randint(0, self.num_modes - 1)
            else:
                # Mode transition: 70% stay, 30% switch
                if rng.random() < 0.7:
                    pass  # keep mode
                else:
                    mode = rng.randint(0, self.num_modes - 1)

            base = mode * self.tokens_per_mode
            # Local Markov: 80% near previous token in same mode
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


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_experiment1(device="cpu"):
    """Run Experiment 1: Small and Medium model comparison."""
    os.makedirs("/mnt/agents/output/resonance/checkpoints", exist_ok=True)
    os.makedirs("/mnt/agents/output/resonance/results", exist_ok=True)
    os.makedirs("/mnt/agents/output/resonance/figures", exist_ok=True)

    # Dataset
    vocab_size = 5000
    seq_len = 64
    train_samples = 3000
    val_samples = 600
    batch_size = 32

    print("=" * 60)
    print(f"Experiment 1: Generating structured synthetic data")
    print(f"  vocab={vocab_size}, seq_len={seq_len}, train={train_samples}, val={val_samples}")
    print("=" * 60)

    train_ds = StructuredSyntheticDataset(vocab_size, seq_len, train_samples, seed=42)
    val_ds = StructuredSyntheticDataset(vocab_size, seq_len, val_samples, seed=123)
    train_loader = get_dataloader(train_ds, batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size, shuffle=False)

    configs = [
        {"name": "small", "d_model": 128, "nhead": 4, "layers": 3, "ff": 512, "epochs": 5},
        {"name": "medium", "d_model": 256, "nhead": 4, "layers": 4, "ff": 1024, "epochs": 3},
    ]

    results = {}

    for cfg in configs:
        for variant in ["standard", "resonance"]:
            model_name = f"exp1_{cfg['name']}_{variant}"
            print(f"\n--- Training {model_name} ---")
            t_start = time.time()

            if variant == "standard":
                model = StandardTransformer(
                    vocab_size=vocab_size,
                    d_model=cfg["d_model"],
                    nhead=cfg["nhead"],
                    num_layers=cfg["layers"],
                    dim_feedforward=cfg["ff"],
                    max_seq_len=seq_len,
                )
            else:
                model = ResonanceTransformer(
                    vocab_size=vocab_size,
                    d_model=cfg["d_model"],
                    nhead=cfg["nhead"],
                    num_layers=cfg["layers"],
                    dim_feedforward=cfg["ff"],
                    max_seq_len=seq_len,
                )

            n_params = count_params(model)
            print(f"  Params: {n_params:,}")

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=cfg["epochs"],
                lr=3e-4,
                weight_decay=0.01,
                max_grad_norm=1.0,
                checkpoint_dir="/mnt/agents/output/resonance/checkpoints",
                model_name=model_name,
                log_interval=25,
            )

            # Full validation metrics
            val_ppl = compute_perplexity(model, val_loader, device=device, max_batches=None)
            val_acc1 = compute_accuracy_at_k(model, val_loader, k=1, device=device, max_batches=None)
            val_acc5 = compute_accuracy_at_k(model, val_loader, k=5, device=device, max_batches=None)

            results[model_name] = {
                "config": cfg,
                "variant": variant,
                "params": n_params,
                "history": history,
                "final_val_ppl": val_ppl,
                "final_val_acc@1": val_acc1,
                "final_val_acc@5": val_acc5,
                "train_time_sec": time.time() - t_start,
            }

            print(f"  Final val_ppl={val_ppl:.2f} | val_acc@1={val_acc1:.4f} | val_acc@5={val_acc5:.4f} | time={time.time()-t_start:.1f}s")

    # Save results
    with open("/mnt/agents/output/resonance/results/exp1_real.json", "w") as f:
        json.dump(results, f, indent=2)

    # Simple perturbation test on best model
    print("\n--- Perturbation Test ---")
    best_name = min(results, key=lambda k: results[k]["final_val_ppl"])
    print(f"Best model: {best_name}")
    best_cfg = results[best_name]["config"]
    best_variant = results[best_name]["variant"]

    if best_variant == "standard":
        best_model = StandardTransformer(vocab_size=vocab_size, d_model=best_cfg["d_model"],
                                          nhead=best_cfg["nhead"], num_layers=best_cfg["layers"],
                                          dim_feedforward=best_cfg["ff"], max_seq_len=seq_len)
    else:
        best_model = ResonanceTransformer(vocab_size=vocab_size, d_model=best_cfg["d_model"],
                                           nhead=best_cfg["nhead"], num_layers=best_cfg["layers"],
                                           dim_feedforward=best_cfg["ff"], max_seq_len=seq_len)

    ckpt = torch.load(f"/mnt/agents/output/resonance/checkpoints/{best_name}.pt", map_location=device)
    best_model.load_state_dict(ckpt["model_state_dict"])
    best_model.to(device)
    best_model.eval()

    # Perturbation: random token flip at position 16
    perturb_results = {}
    with torch.no_grad():
        x, y = next(iter(val_loader))
        x, y = x.to(device), y.to(device)
        logits_orig = best_model(x)

        x_pert = x.clone()
        x_pert[:, 16] = torch.randint(0, vocab_size, (x.size(0),), device=device)
        logits_pert = best_model(x_pert)

        # Compare KL divergence of predictions at position 17 (affected by position 16)
        kl = nn.functional.kl_div(
            nn.functional.log_softmax(logits_pert[:, 17], dim=-1),
            nn.functional.softmax(logits_orig[:, 17], dim=-1),
            reduction="batchmean"
        ).item()
        perturb_results["kl_divergence_at_pos17"] = kl
        perturb_results["mean_abs_diff"] = (logits_pert[:, 17] - logits_orig[:, 17]).abs().mean().item()

    results["perturbation_test"] = perturb_results
    print(f"  Perturbation KL div: {kl:.4f}")

    with open("/mnt/agents/output/resonance/results/exp1_real.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nExperiment 1 complete. Results saved to results/exp1_real.json")
    return results


def run_experiment2(device="cpu"):
    """Run Experiment 2: Mini Scale Study."""
    os.makedirs("/mnt/agents/output/resonance/checkpoints", exist_ok=True)
    os.makedirs("/mnt/agents/output/resonance/results", exist_ok=True)

    vocab_size = 5000
    seq_len = 64
    train_samples = 1000
    val_samples = 200
    batch_size = 32

    print("=" * 60)
    print(f"Experiment 2: Scaling Study")
    print(f"  vocab={vocab_size}, seq_len={seq_len}, train={train_samples}, val={val_samples}")
    print("=" * 60)

    train_ds = StructuredSyntheticDataset(vocab_size, seq_len, train_samples, seed=42)
    val_ds = StructuredSyntheticDataset(vocab_size, seq_len, val_samples, seed=123)
    train_loader = get_dataloader(train_ds, batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size, shuffle=False)

    scales = [
        {"name": "1x", "d_model": 128, "nhead": 4, "layers": 2, "ff": 512},
        {"name": "2x", "d_model": 128, "nhead": 4, "layers": 4, "ff": 512},
        {"name": "4x", "d_model": 256, "nhead": 4, "layers": 4, "ff": 1024},
        {"name": "8x", "d_model": 256, "nhead": 4, "layers": 8, "ff": 1024},
    ]

    results = {}
    epochs = 2  # Reduced for compute

    for scale in scales:
        for variant in ["standard", "resonance"]:
            model_name = f"exp2_{scale['name']}_{variant}"
            print(f"\n--- Training {model_name} ---")
            t_start = time.time()

            if variant == "standard":
                model = StandardTransformer(
                    vocab_size=vocab_size, d_model=scale["d_model"], nhead=scale["nhead"],
                    num_layers=scale["layers"], dim_feedforward=scale["ff"], max_seq_len=seq_len,
                )
            else:
                model = ResonanceTransformer(
                    vocab_size=vocab_size, d_model=scale["d_model"], nhead=scale["nhead"],
                    num_layers=scale["layers"], dim_feedforward=scale["ff"], max_seq_len=seq_len,
                )

            n_params = count_params(model)
            history = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, num_epochs=epochs, lr=3e-4, weight_decay=0.01,
                max_grad_norm=1.0,
                checkpoint_dir="/mnt/agents/output/resonance/checkpoints",
                model_name=model_name, log_interval=10,
            )

            val_ppl = compute_perplexity(model, val_loader, device=device, max_batches=None)
            val_acc1 = compute_accuracy_at_k(model, val_loader, k=1, device=device, max_batches=None)

            results[model_name] = {
                "scale": scale["name"],
                "variant": variant,
                "params": n_params,
                "d_model": scale["d_model"],
                "layers": scale["layers"],
                "history": history,
                "final_val_ppl": val_ppl,
                "final_val_acc@1": val_acc1,
                "train_time_sec": time.time() - t_start,
            }
            print(f"  Params: {n_params:,} | PPL: {val_ppl:.2f} | Acc@1: {val_acc1:.4f} | Time: {time.time()-t_start:.1f}s")

    with open("/mnt/agents/output/resonance/results/exp2_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nExperiment 2 complete. Results saved to results/exp2_scaling.json")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], default=1, help="Which experiment to run")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(4)

    if args.exp == 1:
        run_experiment1(device=args.device)
    elif args.exp == 2:
        run_experiment2(device=args.device)
    else:
        run_experiment1(device=args.device)
        run_experiment2(device=args.device)
