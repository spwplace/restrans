"""
Training utilities for Resonance Transformer.
"""
import os
import math
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any


def compute_perplexity(model: nn.Module, dataloader: DataLoader, device: str = "cpu", max_batches: Optional[int] = None) -> float:
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
            count += 1
            if max_batches is not None and count >= max_batches:
                break
    avg_nll = total_loss / total_tokens
    ppl = math.exp(avg_nll)
    return ppl


def compute_accuracy_at_k(model: nn.Module, dataloader: DataLoader, k: int = 1, device: str = "cpu", max_batches: Optional[int] = None) -> float:
    """Compute top-k accuracy on next-token prediction."""
    model.eval()
    correct = 0
    total = 0
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # top-k prediction
            _, top_k = torch.topk(logits, k, dim=-1)
            correct += (top_k == y.unsqueeze(-1)).any(dim=-1).sum().item()
            total += y.numel()
            count += 1
            if max_batches is not None and count >= max_batches:
                break
    return correct / total if total > 0 else 0.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
    num_epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    checkpoint_dir: str = "./checkpoints",
    model_name: str = "model",
    log_interval: int = 50,
) -> Dict[str, Any]:
    """Train a model and return training history."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    history = {"train_loss": [], "val_ppl": [], "val_acc@1": [], "epochs": []}
    best_ppl = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if batch_idx % log_interval == 0:
                print(f"  [epoch {epoch+1}/{num_epochs} batch {batch_idx}] loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.6f}")

        avg_train_loss = epoch_loss / epoch_steps
        val_ppl = compute_perplexity(model, val_loader, device=device, max_batches=20)
        val_acc1 = compute_accuracy_at_k(model, val_loader, k=1, device=device, max_batches=20)

        history["train_loss"].append(avg_train_loss)
        history["val_ppl"].append(val_ppl)
        history["val_acc@1"].append(val_acc1)
        history["epochs"].append(epoch + 1)

        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={avg_train_loss:.4f} | val_ppl={val_ppl:.2f} | val_acc@1={val_acc1:.4f} | time={time.time()-t0:.1f}s")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_ppl": best_ppl,
                "history": history,
            }, ckpt_path)
            print(f"  -> Saved checkpoint to {ckpt_path}")

    return history


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return ckpt
