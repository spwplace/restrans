#!/usr/bin/env python3
"""
Unified Training Harness for Resonance Transformers
====================================================

Supports three training modes:
  1. standard   -- Vanilla transformer on structured synthetic data
  2. resonance  -- Resonance transformer on structured synthetic data
  3. synthetic  -- Contrastive proof-walk training on lambda calculus

Optimized for Apple Silicon (MPS) with auto-detection and scale presets.

Usage:
    uv run python resonance/train.py --mode resonance --scale small --epochs 20
    uv run python resonance/train.py --mode synthetic --scale medium --crawl_mode hybrid
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure package imports work when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

from resonance.device import get_device, set_seed, enable_deterministic
from resonance.config import StandardConfig, ResonanceConfig, count_params
from resonance.models import StandardTransformer, ResonanceTransformer
from resonance.training import train_model

# ---------------------------------------------------------------------------
# Synthetic pipeline imports (best-effort)
# ---------------------------------------------------------------------------
try:
    from synthetic.proof_walk_generator import ProofWalkDataset, CrawlMode
    from synthetic.mutation_engine import BisimilarMutation, training_schedule
    from synthetic.contrastive_trainer import (
        MinimalResonanceTransformer,
        ProgramTokenizer,
        DualContrastiveLoss,
    )
    SYNTHETIC_AVAILABLE = True
except Exception as e:
    SYNTHETIC_AVAILABLE = False
    _SYNTHETIC_ERR = str(e)


# =============================================================================
# 1. Structured Synthetic Dataset (for standard/resonance modes)
# =============================================================================

class StructuredSyntheticDataset(Dataset):
    """Structured synthetic language modelling dataset with rhythmic modes."""

    def __init__(
        self,
        vocab_size: int = 5000,
        seq_len: int = 64,
        num_samples: int = 3000,
        num_modes: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_modes = num_modes
        self.tokens_per_mode = vocab_size // num_modes
        self._seed = seed
        if seed is not None:
            import random
            random.seed(seed)
            torch.manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import random
        rng = random.Random(idx + (self._seed or 42))
        x = torch.zeros(self.seq_len, dtype=torch.long)
        mode = rng.randint(0, self.num_modes - 1)
        x[0] = mode * self.tokens_per_mode + rng.randint(0, self.tokens_per_mode - 1)
        for t in range(1, self.seq_len):
            if t % 8 == 0 and t >= 8:
                prev_mode = min((x[t - 8].item() // self.tokens_per_mode), self.num_modes - 1)
                mode = prev_mode if rng.random() < 0.7 else rng.randint(0, self.num_modes - 1)
            else:
                mode = mode if rng.random() < 0.7 else rng.randint(0, self.num_modes - 1)
            base = mode * self.tokens_per_mode
            if rng.random() < 0.8:
                prev_in_mode = x[t - 1].item() - base
                token = base + ((prev_in_mode + rng.randint(-3, 3)) % self.tokens_per_mode)
            else:
                token = base + rng.randint(0, self.tokens_per_mode - 1)
            x[t] = max(0, min(token, self.vocab_size - 1))
        return {"input_ids": x, "labels": x.clone()}


# =============================================================================
# 2. Scale Presets
# =============================================================================

SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "epochs": 5,
        "train_samples": 1000,
        "val_samples": 200,
        "batch_size": 16,
        "seq_len": 64,
        "embed_dim": 128,
        "n_layers": 2,
        "n_heads": 4,
        "ff_dim": 512,
        "learning_rate": 3e-4,
    },
    "small": {
        "epochs": 20,
        "train_samples": 10000,
        "val_samples": 2000,
        "batch_size": 32,
        "seq_len": 128,
        "embed_dim": 256,
        "n_layers": 4,
        "n_heads": 8,
        "ff_dim": 1024,
        "learning_rate": 3e-4,
    },
    "medium": {
        "epochs": 30,
        "train_samples": 50000,
        "val_samples": 5000,
        "batch_size": 32,
        "seq_len": 256,
        "embed_dim": 512,
        "n_layers": 6,
        "n_heads": 8,
        "ff_dim": 2048,
        "learning_rate": 1e-4,
    },
    "large": {
        "epochs": 50,
        "train_samples": 200000,
        "val_samples": 10000,
        "batch_size": 16,  # smaller batch for memory
        "seq_len": 256,
        "embed_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "ff_dim": 3072,
        "learning_rate": 5e-5,
    },
}

SYNTHETIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "epochs": 5,
        "groups": 200,
        "programs_per_group": 4,
        "batch_size": 8,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 2,
        "max_length": 64,
        "max_term_depth": 4,
        "max_term_size": 12,
        "max_mutations": 100,
        "lr": 3e-4,
    },
    "small": {
        "epochs": 20,
        "groups": 2000,
        "programs_per_group": 8,
        "batch_size": 16,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "max_length": 128,
        "max_term_depth": 5,
        "max_term_size": 16,
        "max_mutations": 1000,
        "lr": 3e-4,
    },
    "medium": {
        "epochs": 30,
        "groups": 10000,
        "programs_per_group": 8,
        "batch_size": 32,
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "max_length": 128,
        "max_term_depth": 6,
        "max_term_size": 20,
        "max_mutations": 5000,
        "lr": 1e-4,
    },
    "large": {
        "epochs": 50,
        "groups": 50000,
        "programs_per_group": 8,
        "batch_size": 32,
        "d_model": 512,
        "n_layers": 8,
        "n_heads": 8,
        "max_length": 256,
        "max_term_depth": 6,
        "max_term_size": 24,
        "max_mutations": 10000,
        "lr": 5e-5,
    },
}


# =============================================================================
# 3. CLI
# =============================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Resonance Transformers")
    parser.add_argument("--mode", choices=["standard", "resonance", "synthetic"], required=True)
    parser.add_argument("--scale", default="small", choices=list(SCALE_PRESETS.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--crawl_mode", default="hybrid", choices=["structured", "random", "hybrid"])
    parser.add_argument("--mutate", action="store_true", default=True, help="Apply bisimilar mutations")
    parser.add_argument("--no_mutate", action="store_true", help="Disable mutations")
    parser.add_argument("--save_interval", type=int, default=5)
    return parser.parse_args()


# =============================================================================
# 4. Training -- Standard / Resonance
# =============================================================================

def train_standard_or_resonance(args: argparse.Namespace) -> None:
    preset = SCALE_PRESETS[args.scale]
    epochs = args.epochs or preset["epochs"]
    batch_size = args.batch_size or preset["batch_size"]
    device = get_device(args.device)

    out_dir = Path(args.output_dir or f"resonance/outputs/{args.mode}_{args.scale}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    config_dict = {
        "mode": args.mode,
        "scale": args.scale,
        "seed": args.seed,
        "device": str(device),
        "preset": preset,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 60)
    print(f"Mode: {args.mode.upper()} | Scale: {args.scale}")
    print(f"Device: {device}")
    print("=" * 60)

    vocab_size = 5000
    train_ds = StructuredSyntheticDataset(
        vocab_size=vocab_size,
        seq_len=preset["seq_len"],
        num_samples=preset["train_samples"],
        seed=args.seed,
    )
    val_ds = StructuredSyntheticDataset(
        vocab_size=vocab_size,
        seq_len=preset["seq_len"],
        num_samples=preset["val_samples"],
        seed=args.seed + 1,
    )

    if args.mode == "standard":
        config = StandardConfig(
            vocab_size=vocab_size,
            max_seq_len=preset["seq_len"],
            embed_dim=preset["embed_dim"],
            n_layers=preset["n_layers"],
            n_heads=preset["n_heads"],
            ff_dim=preset["ff_dim"],
            batch_size=batch_size,
            learning_rate=preset["learning_rate"],
        )
        model = StandardTransformer(config)
    else:
        config = ResonanceConfig(
            vocab_size=vocab_size,
            max_seq_len=preset["seq_len"],
            embed_dim=preset["embed_dim"],
            n_layers=preset["n_layers"],
            n_heads=preset["n_heads"],
            ff_dim=preset["ff_dim"],
            n_frequencies=32,
            batch_size=batch_size,
            learning_rate=preset["learning_rate"],
            phonetic_init=False,
        )
        model = ResonanceTransformer(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Config: {config}")

    history = train_model(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        n_epochs=epochs,
        device=device,
        log_interval=max(1, len(train_ds) // batch_size // 10),
    )

    # Save final
    final_path = out_dir / "checkpoints" / "final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "history": history,
        "args": vars(args),
    }, final_path)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved to {out_dir}")


# =============================================================================
# 5. Training -- Synthetic Proof-Walk
# =============================================================================

def train_synthetic_mode(args: argparse.Namespace) -> None:
    if not SYNTHETIC_AVAILABLE:
        raise RuntimeError(f"Synthetic pipeline not available: {_SYNTHETIC_ERR}")

    preset = SYNTHETIC_PRESETS[args.scale]
    epochs = args.epochs or preset["epochs"]
    batch_size = args.batch_size or preset["batch_size"]
    device = get_device(args.device)
    mutate = args.mutate and not args.no_mutate

    out_dir = Path(args.output_dir or f"resonance/outputs/synthetic_{args.scale}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    crawl_mode = CrawlMode[args.crawl_mode.upper()]

    config_dict = {
        "mode": "synthetic",
        "scale": args.scale,
        "seed": args.seed,
        "device": str(device),
        "crawl_mode": crawl_mode.name,
        "mutate": mutate,
        "preset": preset,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 60)
    print(f"Mode: SYNTHETIC | Scale: {args.scale} | Crawl: {crawl_mode.name}")
    print(f"Device: {device}")
    print("=" * 60)

    # Dataset
    base_dataset = ProofWalkDataset(
        n_groups=preset["groups"],
        programs_per_group=preset["programs_per_group"],
        max_term_depth=preset["max_term_depth"],
        max_term_size=preset["max_term_size"],
        generator_seed=args.seed,
        crawl_mode=crawl_mode,
        structured_ratio=0.5,
    )
    print(f"Dataset: {len(base_dataset)} groups, {preset['programs_per_group']} programs/group")

    # Mutations
    mutation_engine = BisimilarMutation(seed=args.seed, verify=True)
    if mutate:
        from synthetic.proof_walk_generator import MutatedProofWalkDataset
        dataset = MutatedProofWalkDataset(base_dataset, mutation_engine)
    else:
        dataset = base_dataset

    # Model
    tokenizer = ProgramTokenizer(max_length=preset["max_length"])
    model = MinimalResonanceTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=preset["d_model"],
        n_layers=preset["n_layers"],
        n_heads=preset["n_heads"],
        d_semantic=preset["d_model"] // 2,
        d_phase=preset["d_model"] // 2,
        max_length=preset["max_length"],
    )
    model = model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=preset["lr"])
    contrastive_loss_fn = DualContrastiveLoss(temperature=0.07, loss_type="infonce")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    history: Dict[str, List[float]] = {
        "train_lm": [],
        "train_ctr": [],
        "train_total": [],
        "mutations": [],
    }

    for epoch in range(epochs):
        # Scheduled mutations
        n_mutations = 0
        if mutate:
            n_mutations = training_schedule(
                epoch, epochs, schedule="linear", max_mutations=preset["max_mutations"]
            )
            if n_mutations > 0 and hasattr(dataset, "mutate_all"):
                total_edits = dataset.mutate_all(n_mutations)
                print(f"  Applied {n_mutations} mutations/group ({total_edits} total edits)")

        # Train
        model.train()
        epoch_lm = 0.0
        epoch_ctr = 0.0
        epoch_total = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            programs_batch = batch["programs"]
            all_programs: List[str] = []
            group_sizes: List[int] = []
            for group_progs in programs_batch:
                all_programs.extend(group_progs)
                group_sizes.append(len(group_progs))
            if not all_programs:
                continue

            tokens = tokenizer.batch_encode(all_programs, device=str(device))
            logits, sem_embeds, phase_embeds = model(tokens)

            targets = tokens[:, 1:].contiguous()
            lm_logits = logits[:, :-1, :].contiguous()
            lm_loss = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_id,
            )

            total = sum(group_sizes)
            positive_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            offset = 0
            for size in group_sizes:
                positive_mask[offset : offset + size, offset : offset + size] = True
                offset += size
            negative_mask = ~positive_mask
            diag = torch.arange(total, device=device)
            negative_mask[diag, diag] = False

            ctr_loss, _ = contrastive_loss_fn(
                sem_embeds, phase_embeds, positive_mask, negative_mask
            )

            total_loss = lm_loss + ctr_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_lm += lm_loss.item()
            epoch_ctr += ctr_loss.item()
            epoch_total += total_loss.item()
            n_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  [E{epoch+1} B{batch_idx+1}] "
                    f"LM={lm_loss.item():.4f} CTR={ctr_loss.item():.4f} "
                    f"Total={total_loss.item():.4f}"
                )

        history["train_lm"].append(epoch_lm / max(n_batches, 1))
        history["train_ctr"].append(epoch_ctr / max(n_batches, 1))
        history["mutations"].append(n_mutations)

        print(
            f"\n=== Epoch {epoch+1}/{epochs} ==="
            f"\n  LM={history['train_lm'][-1]:.4f} "
            f"CTR={history['train_ctr'][-1]:.4f} "
            f"Total={(epoch_lm + epoch_ctr) / max(n_batches, 1):.4f}"
        )

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = out_dir / "checkpoints" / f"epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    final_path = out_dir / "checkpoints" / "final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "history": history,
        "args": vars(args),
    }, final_path)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved to {out_dir}")


# =============================================================================
# 6. Main
# =============================================================================

def main() -> None:
    args = get_args()
    set_seed(args.seed)
    enable_deterministic(True)

    if args.mode in ("standard", "resonance"):
        train_standard_or_resonance(args)
    elif args.mode == "synthetic":
        train_synthetic_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
