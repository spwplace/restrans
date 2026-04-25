"""
train_single.py
===============
Train a single model configuration and save checkpoint + metrics.
Usage:
  python train_single.py --model standard --name exp1_small_standard --d_model 128 --layers 3 --epochs 5
"""
import os
import sys
import time
import json
import torch

sys.path.insert(0, '/mnt/agents/output/resonance')
from models import StandardTransformer, ResonanceTransformer
from utils.training import train_model, compute_perplexity, compute_accuracy_at_k
from torch.utils.data import Dataset, DataLoader


class StructuredSyntheticDataset(Dataset):
    def __init__(self, vocab_size=5000, seq_len=64, num_samples=3000, num_modes=10, seed=None):
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


def get_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["standard", "resonance"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_samples", type=int, default=3000)
    parser.add_argument("--val_samples", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(4)
    os.makedirs("/mnt/agents/output/resonance/checkpoints", exist_ok=True)
    os.makedirs("/mnt/agents/output/resonance/results", exist_ok=True)

    vocab_size = 5000
    train_ds = StructuredSyntheticDataset(vocab_size, args.seq_len, args.train_samples, seed=42)
    val_ds = StructuredSyntheticDataset(vocab_size, args.seq_len, args.val_samples, seed=123)
    train_loader = get_dataloader(train_ds, args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, args.batch_size, shuffle=False)

    if args.model == "standard":
        model = StandardTransformer(vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
                                     num_layers=args.layers, dim_feedforward=args.ff, max_seq_len=args.seq_len)
    else:
        model = ResonanceTransformer(vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
                                      num_layers=args.layers, dim_feedforward=args.ff, max_seq_len=args.seq_len)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.name}] Params: {n_params:,} | Training for {args.epochs} epochs")
    t0 = time.time()

    history = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=args.device, num_epochs=args.epochs, lr=args.lr, weight_decay=0.01,
        max_grad_norm=1.0,
        checkpoint_dir="/mnt/agents/output/resonance/checkpoints",
        model_name=args.name, log_interval=50,
    )

    val_ppl = compute_perplexity(model, val_loader, device=args.device, max_batches=None)
    val_acc1 = compute_accuracy_at_k(model, val_loader, k=1, device=args.device, max_batches=None)
    val_acc5 = compute_accuracy_at_k(model, val_loader, k=5, device=args.device, max_batches=None)

    result = {
        "name": args.name,
        "model": args.model,
        "d_model": args.d_model,
        "layers": args.layers,
        "ff": args.ff,
        "epochs": args.epochs,
        "params": n_params,
        "history": history,
        "final_val_ppl": val_ppl,
        "final_val_acc@1": val_acc1,
        "final_val_acc@5": val_acc5,
        "train_time_sec": time.time() - t0,
    }

    out_path = f"/mnt/agents/output/resonance/results/{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{args.name}] Done. PPL={val_ppl:.2f} Acc@1={val_acc1:.4f} Time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
