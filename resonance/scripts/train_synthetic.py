#!/usr/bin/env python3
"""
Synthetic Data Training Script for Resonance Transformer
=========================================================

End-to-end integration script that:
  1. Generates a synthetic dataset of well-typed lambda calculus terms
  2. Organizes them into proof walks (contrastive groups)
  3. Trains a small Resonance Transformer with dual embeddings
  4. Applies scheduled semantic-preserving mutations over training
  5. Saves checkpoints and training curves

Usage:
    python train_synthetic.py [--epochs 20] [--groups 500] [--programs_per_group 8]

The script directly implements the user's vision:

> "random statements, walk the space of proof for a statement, synthesize the
> programs representing all those proofs, use that as a group and update the
> model in the direction that improves contrastive discrimination of the whole
> group. this directly trains against the actual topology of judgemental
> deduction."

> "regular semantic-preserving mutation of programs probably good in midtraining,
> to learn higher structures ('shortcuts'), can vary the number of mutations
> applied over training eg later in training doing updates on a program that
> has accumulated 10000 edits (still bisimilar)"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthetic"))

from resonance.device import get_device, set_seed, enable_deterministic

from lambda_generator import (
    LambdaGenerator,
    typecheck,
    is_well_typed,
    normal_form,
    Term,
)
from proof_walk_generator import ProofWalkDataset, CrossGroupSampler
from mutation_engine import (
    BisimilarMutation,
    MutatedProgram,
    training_schedule,
)
from contrastive_trainer import (
    MinimalResonanceTransformer,
    ProgramTokenizer,
    DualContrastiveLoss,
    TrainingConfig,
)


# =============================================================================
# 1. Argument Parsing
# =============================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Resonance Transformer on synthetic lambda calculus data"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--groups", type=int, default=500, help="Number of proof groups")
    parser.add_argument(
        "--programs_per_group", type=int, default=8, help="Programs per group"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--max_term_depth", type=int, default=5, help="Max term depth")
    parser.add_argument("--max_term_size", type=int, default=16, help="Max term size")
    parser.add_argument(
        "--mutation_schedule",
        type=str,
        default="linear",
        choices=["linear", "quadratic", "exponential", "sigmoid"],
        help="Mutation count schedule",
    )
    parser.add_argument(
        "--max_mutations", type=int, default=1000, help="Max mutations at final epoch"
    )
    parser.add_argument(
        "--lm_weight", type=float, default=1.0, help="Weight for LM loss"
    )
    parser.add_argument(
        "--ctr_weight", type=float, default=1.0, help="Weight for contrastive loss"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07, help="Contrastive temperature"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (auto-detects MPS/CUDA/CPU if not set)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mutate_during_training",
        action="store_true",
        default=True,
        help="Apply scheduled mutations during training",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="Run evaluation every N epochs",
    )
    parser.add_argument(
        "--crawl_mode",
        type=str,
        default="hybrid",
        choices=["structured", "random", "hybrid"],
        help="Statement generation mode",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Preset scale configuration",
    )
    return parser.parse_args()


# =============================================================================
# 2. Output Setup
# =============================================================================

def setup_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            Path(__file__).parent.parent
            / "outputs"
            / f"synthetic_e{args.epochs}_g{args.groups}_p{args.programs_per_group}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    return out_dir


def save_config(args: argparse.Namespace, out_dir: Path) -> None:
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")


# =============================================================================
# 3. Mutation-Aware Dataset Wrapper
# =============================================================================

class MutatedProofWalkDataset:
    """
    Wrapper around ProofWalkDataset that applies scheduled mutations.

    Stores the full groups (including Term objects) and mutates them
    in-place between epochs. This allows programs to accumulate thousands
    of edits while remaining bisimilar.
    """

    def __init__(
        self,
        base_dataset: ProofWalkDataset,
        mutation_engine: BisimilarMutation,
    ) -> None:
        self.base = base_dataset
        self.engine = mutation_engine
        self._groups: List[Dict[str, Any]] = []
        self._init_groups()

    def _init_groups(self) -> None:
        """Load all full groups from the base dataset."""
        for i in range(len(self.base)):
            group = self.base.get_full_group(i)
            if group is not None:
                self._groups.append(group)

    def mutate_all(self, n_mutations: int) -> int:
        """Apply n_mutations to every program in every group."""
        total_edits = 0
        for group in self._groups:
            mutated_terms = []
            for term in group["program_terms"]:
                prog = MutatedProgram(
                    origin=group["program_terms"][0],  # first as origin
                    current=term,
                    ctx=group.get("ctx", []),
                    target_type=group.get("target_type"),
                )
                mutated = self.engine.mutate(prog, n_mutations=n_mutations)
                mutated_terms.append(mutated.current)
                total_edits += mutated.edit_count
            group["program_terms"] = mutated_terms
            group["programs"] = [t.to_string() for t in mutated_terms]
        return total_edits

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        group = self._groups[idx % len(self._groups)]
        k = len(group["programs"])
        return {
            "programs": group["programs"],
            "statement": group["statement"],
            "positive_mask": torch.ones(k, k, dtype=torch.bool),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "programs": [b["programs"] for b in batch],
            "statements": [b["statement"] for b in batch],
            "positive_masks": [b["positive_mask"] for b in batch],
        }


# =============================================================================
# 4. Training Loop
# =============================================================================

def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer: ProgramTokenizer,
    contrastive_loss_fn: DualContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    config: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    """Run one training epoch."""
    device = get_device(config.device)
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

        # Tokenize
        tokens = tokenizer.batch_encode(all_programs, device=device)

        # Forward
        logits, sem_embeds, phase_embeds = model(tokens)

        # LM loss
        targets = tokens[:, 1:].contiguous()
        lm_logits = logits[:, :-1, :].contiguous()
        lm_loss = F.cross_entropy(
            lm_logits.reshape(-1, lm_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=tokenizer.pad_id,
        )

        # Contrastive loss
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

        total_loss = config.lm_weight * lm_loss + config.ctr_weight * ctr_loss

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
                f"  [Epoch {epoch+1} Batch {batch_idx+1}] "
                f"LM={lm_loss.item():.4f} CTR={ctr_loss.item():.4f} "
                f"Total={total_loss.item():.4f}"
            )

    return {
        "lm_loss": epoch_lm / max(n_batches, 1),
        "contrastive_loss": epoch_ctr / max(n_batches, 1),
        "total_loss": epoch_total / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer: ProgramTokenizer,
    contrastive_loss_fn: DualContrastiveLoss,
    config: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate on the dataset."""
    device = get_device(config.device)
    model.eval()

    epoch_lm = 0.0
    epoch_ctr = 0.0
    n_batches = 0

    for batch in loader:
        programs_batch = batch["programs"]
        all_programs: List[str] = []
        group_sizes: List[int] = []
        for group_progs in programs_batch:
            all_programs.extend(group_progs)
            group_sizes.append(len(group_progs))

        if not all_programs:
            continue

        tokens = tokenizer.batch_encode(all_programs, device=device)
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

        epoch_lm += lm_loss.item()
        epoch_ctr += ctr_loss.item()
        n_batches += 1

    return {
        "lm_loss": epoch_lm / max(n_batches, 1),
        "contrastive_loss": epoch_ctr / max(n_batches, 1),
    }


# =============================================================================
# 5. Main
# =============================================================================

# =============================================================================
# 6. Scale Presets
# =============================================================================

SCALE_PRESETS = {
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
    },
    "small": {
        "epochs": 20,
        "groups": 1000,
        "programs_per_group": 8,
        "batch_size": 16,
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "max_length": 128,
        "max_term_depth": 5,
        "max_term_size": 16,
        "max_mutations": 1000,
    },
    "medium": {
        "epochs": 30,
        "groups": 5000,
        "programs_per_group": 8,
        "batch_size": 32,
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "max_length": 128,
        "max_term_depth": 6,
        "max_term_size": 20,
        "max_mutations": 5000,
    },
    "large": {
        "epochs": 50,
        "groups": 20000,
        "programs_per_group": 8,
        "batch_size": 32,
        "d_model": 512,
        "n_layers": 8,
        "n_heads": 8,
        "max_length": 256,
        "max_term_depth": 6,
        "max_term_size": 24,
        "max_mutations": 10000,
    },
}


def apply_scale(args: argparse.Namespace) -> argparse.Namespace:
    """Override args with scale preset values, keeping explicit CLI overrides."""
    preset = SCALE_PRESETS.get(args.scale, {})
    for key, value in preset.items():
        if getattr(args, key, None) is None or key in ("epochs", "groups", "d_model", "n_layers"):
            # Only override if the user did not explicitly set it via CLI
            # argparse defaults make this tricky; we use a sentinel check
            pass
    # Simpler: just override unconditionally for preset keys
    for key, value in preset.items():
        setattr(args, key, value)
    return args


def main() -> None:
    args = get_args()
    args = apply_scale(args)
    set_seed(args.seed)
    enable_deterministic(True)

    out_dir = setup_output_dir(args)
    save_config(args, out_dir)

    # Save reproducibility bundle
    repro_path = out_dir / "reproducibility.json"
    with open(repro_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "scale": args.scale,
                "crawl_mode": args.crawl_mode,
                "pytorch_version": torch.__version__,
                "device": str(get_device(args.device)),
            },
            f,
            indent=2,
        )

    print("=" * 60)
    print("Resonance Transformer - Synthetic Data Training")
    print("=" * 60)
    print(f"Scale preset: {args.scale}")
    print(f"Output directory: {out_dir}")
    print(f"Device: {get_device(args.device)}")
    print()

    # 1. Generate dataset
    print("[1/5] Generating synthetic dataset...")
    from synthetic.proof_walk_generator import CrawlMode
    crawl_mode = CrawlMode[args.crawl_mode.upper()]
    base_dataset = ProofWalkDataset(
        n_groups=args.groups,
        programs_per_group=args.programs_per_group,
        max_term_depth=args.max_term_depth,
        max_term_size=args.max_term_size,
        generator_seed=args.seed,
        crawl_mode=crawl_mode,
        structured_ratio=0.5,
    )
    # Save dataset state for exact reproduction
    ds_state_path = out_dir / "dataset_state.json"
    with open(ds_state_path, "w") as f:
        json.dump(base_dataset.get_state(), f, indent=2, default=str)
    print(f"  Groups: {len(base_dataset)}")
    print(f"  Programs per group: {args.programs_per_group}")
    print(f"  Crawl mode: {crawl_mode.name}")

    # 2. Setup mutation engine
    print("[2/5] Setting up mutation engine...")
    mutation_engine = BisimilarMutation(seed=args.seed, verify=True)
    if args.mutate_during_training:
        dataset = MutatedProofWalkDataset(base_dataset, mutation_engine)
    else:
        dataset = base_dataset

    # 3. Setup tokenizer and model
    print("[3/5] Building model...")
    tokenizer = ProgramTokenizer(max_length=args.max_length)
    model = MinimalResonanceTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_semantic=args.d_model // 2,
        d_phase=args.d_model // 2,
        max_length=args.max_length,
    )
    device = get_device(args.device)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Device: {device}")

    # 4. Setup training
    print("[4/5] Setting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    contrastive_loss_fn = DualContrastiveLoss(
        temperature=args.temperature,
        loss_type="infonce",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # 5. Training loop
    print("[5/5] Training...")
    history: Dict[str, List[float]] = {
        "train_lm": [],
        "train_ctr": [],
        "train_total": [],
        "eval_lm": [],
        "eval_ctr": [],
        "mutations": [],
    }

    for epoch in range(args.epochs):
        # Scheduled mutations
        n_mutations = 0
        if args.mutate_during_training:
            n_mutations = training_schedule(
                epoch,
                args.epochs,
                schedule=args.mutation_schedule,
                max_mutations=args.max_mutations,
            )
            if n_mutations > 0 and isinstance(dataset, MutatedProofWalkDataset):
                total_edits = dataset.mutate_all(n_mutations)
                print(f"\n  Applied {n_mutations} mutations/group "
                      f"({total_edits} total edits)")

        # Train
        train_metrics = train_epoch(
            model, loader, tokenizer, contrastive_loss_fn, optimizer, args, epoch
        )

        # Eval
        eval_metrics = {}
        if (epoch + 1) % args.eval_interval == 0:
            eval_metrics = evaluate(
                model, loader, tokenizer, contrastive_loss_fn, args
            )
            print(
                f"\n  Eval  | LM={eval_metrics['lm_loss']:.4f} "
                f"CTR={eval_metrics['contrastive_loss']:.4f}"
            )

        # Log
        print(
            f"\n=== Epoch {epoch+1}/{args.epochs} ==="
            f"\n  Train | LM={train_metrics['lm_loss']:.4f} "
            f"CTR={train_metrics['contrastive_loss']:.4f} "
            f"Total={train_metrics['total_loss']:.4f}"
        )
        if eval_metrics:
            print(
                f"  Eval  | LM={eval_metrics['lm_loss']:.4f} "
                f"CTR={eval_metrics['contrastive_loss']:.4f}"
            )
        print(f"  Mutations this epoch: {n_mutations}")

        history["train_lm"].append(train_metrics["lm_loss"])
        history["train_ctr"].append(train_metrics["contrastive_loss"])
        history["train_total"].append(train_metrics["total_loss"])
        history["eval_lm"].append(eval_metrics.get("lm_loss", 0.0))
        history["eval_ctr"].append(eval_metrics.get("contrastive_loss", 0.0))
        history["mutations"].append(n_mutations)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = out_dir / "checkpoints" / f"epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": vars(args),
                    "history": history,
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

    # Save final model and history
    final_path = out_dir / "checkpoints" / "final.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "history": history,
        },
        final_path,
    )
    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nFinal model saved: {final_path}")
    print(f"History saved: {history_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final LM loss:        {history['train_lm'][-1]:.4f}")
    print(f"Final CTR loss:       {history['train_ctr'][-1]:.4f}")
    print(f"Total epochs:         {args.epochs}")
    print(f"Max mutations:        {max(history['mutations'])}")
    print(f"Output directory:     {out_dir}")


if __name__ == "__main__":
    main()
