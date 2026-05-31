#!/usr/bin/env python3
"""
Web-scale training with GPT-2 BPE on Fineweb-edu.
Streams data from HuggingFace to avoid downloading full dataset.

Usage:
    python train_webscale.py --condition standard_alibi --scale 20M --max_tokens 100M --device mps
    python train_webscale.py --condition phase_dynamic_qk_film --scale 50M --max_tokens 500M --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from resonance.device import get_device, set_seed
from resonance.config import StandardConfig, ResonanceConfig
from resonance.models import StandardTransformer, ResonanceTransformer


# =============================================================================
# 1. GPT-2 BPE Tokenizer
# =============================================================================

class GPT2TokenizerWrapper:
    """Wrapper around HuggingFace GPT-2 tokenizer."""

    def __init__(self, max_length: int = 512, cache_dir: Optional[str] = None) -> None:
        from transformers import GPT2Tokenizer
        self._tok = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
        self._tok.pad_token = self._tok.eos_token
        self.vocab_size = self._tok.vocab_size
        self.pad_id = self._tok.pad_token_id
        self.eos_id = self._tok.eos_token_id
        self.max_length = max_length
        print(f"GPT-2 tokenizer: vocab={self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        # Do NOT truncate here — let the sliding window handle long texts
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


# =============================================================================
# 2. Streaming Dataset
# =============================================================================

class FinewebStreamingDataset(IterableDataset):
    """Stream Fineweb-edu from HuggingFace with optional local buffer."""

    def __init__(
        self,
        tokenizer: GPT2TokenizerWrapper,
        seq_len: int = 512,
        split: str = "train",
        subset: Optional[str] = "sample-10BT",
        max_tokens: Optional[int] = None,
        prefetch_buffer: int = 50000,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.split = split
        self.subset = subset
        self.prefetch_buffer = prefetch_buffer
        self.cache_dir = cache_dir
        self.streaming = streaming
        self._buffer: List[Dict[str, torch.Tensor]] = []
        self._token_count = 0
        self._exhausted = False

    def _load_buffer(self):
        """Prefetch examples into local buffer for fast training."""
        from datasets import load_dataset
        if self._exhausted:
            return

        mode = "HF stream" if self.streaming else "repo-local HF dataset cache"
        print(f"[Data] Loading up to {self.prefetch_buffer} examples from {mode}...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            self.subset,
            streaming=self.streaming,
            split=self.split,
            cache_dir=self.cache_dir,
        )

        count = 0
        for example in ds:
            if count >= self.prefetch_buffer:
                break
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)

            for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                if self.max_tokens and self._token_count >= self.max_tokens:
                    self._exhausted = True
                    break
                chunk = tokens[i : i + self.seq_len + 1]
                if len(chunk) >= 2:
                    self._buffer.append({
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    })
                    self._token_count += len(chunk) - 1
            count += 1
            if self._exhausted:
                break

        print(f"[Data] Buffer loaded: {len(self._buffer)} sequences, {self._token_count/1e6:.1f}M tokens")

    def __iter__(self):
        if not self._buffer:
            self._load_buffer()
        for item in self._buffer:
            yield item

    def __len__(self):
        if not self._buffer:
            self._load_buffer()
        return len(self._buffer)


# =============================================================================
# 3. Model Factory
# =============================================================================

SCALES = {
    "10M": {"d_model": 384, "n_layers": 4, "n_heads": 6, "d_ff": 1024, "dropout": 0.1},
    "20M": {"d_model": 512, "n_layers": 4, "n_heads": 8, "d_ff": 1536, "dropout": 0.1},
    "50M": {"d_model": 768, "n_layers": 6, "n_heads": 12, "d_ff": 2048, "dropout": 0.1},
    "100M": {"d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 3072, "dropout": 0.1},
}


def estimate_params(embed_dim: int, n_layers: int, n_heads: int, ff_dim: int, vocab_size: int = 50257) -> int:
    """Rough parameter count."""
    emb = vocab_size * embed_dim + embed_dim  # token + pos
    per_layer = (
        4 * embed_dim * embed_dim  # QKV + O projections
        + 2 * embed_dim * ff_dim   # FFN
        + 4 * embed_dim            # LayerNorms + biases
    )
    head = embed_dim * vocab_size + vocab_size  # lm_head
    total = emb + n_layers * per_layer + head
    return total


def configure_condition(config, condition: str):
    """Apply condition-specific settings to config."""
    base = condition.removesuffix("_normalized")
    if base in {"phase_dynamic_qk_film", "phase_dynamic_qk_film_alibi"}:
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.phase_update_mode = "mlp"
        config.phase_condition_qk = "film"
        if base.endswith("_alibi"):
            config.attention_variant = "alibi"
    elif base in {"relation_value_qk_film", "relation_value_qk_film_alibi"}:
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.phase_condition_qk = "film"
        config.relation_value_mode = "additive"
        if base.endswith("_alibi"):
            config.attention_variant = "alibi"
    elif base == "phase_stream_only":
        config.use_phase_stream = True
        config.use_resonance_bias = False
    elif base == "resonance_full":
        config.use_phase_stream = True
        config.use_resonance_bias = True
    elif base == "standard_alibi":
        if hasattr(config, "attention_variant"):
            config.attention_variant = "alibi"
    if "normalized" in condition:
        config.normalize_resonance = True
    return config


def is_standard_condition(condition: str) -> bool:
    base = condition.removesuffix("_normalized")
    return base in {"standard", "standard_alibi", "standard_deberta_lite"}


def create_model(condition: str, scale: str, vocab_size: int = 50257, device: torch.device = None):
    """Create model based on condition and scale."""
    cfg = SCALES[scale]
    est_cfg = {
        "embed_dim": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "ff_dim": cfg["d_ff"],
    }
    params = estimate_params(vocab_size=vocab_size, **est_cfg)
    print(f"Scale {scale}: d={cfg['d_model']}, L={cfg['n_layers']}, H={cfg['n_heads']}")
    print(f"Estimated params: {params / 1e6:.1f}M")

    if is_standard_condition(condition):
        config = StandardConfig(
            vocab_size=vocab_size,
            max_seq_len=512,
            embed_dim=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            ff_dim=cfg["d_ff"],
            dropout=cfg["dropout"],
        )
        config = configure_condition(config, condition)
        model = StandardTransformer(config)
    else:
        config = ResonanceConfig(
            vocab_size=vocab_size,
            max_seq_len=512,
            embed_dim=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            ff_dim=cfg["d_ff"],
            dropout=cfg["dropout"],
            n_frequencies=32,
        )
        config = configure_condition(config, condition)
        model = ResonanceTransformer(config)

    if device:
        model = model.to(device)
    return model, config


# =============================================================================
# 4. Training Loop
# =============================================================================

def train_webscale(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_steps: int = 10000,
    eval_every: int = 1000,
    log_every: int = 100,
    lr: float = 3e-4,
    warmup_steps: int = 1000,
    grad_clip: float = 1.0,
    save_path: Optional[Path] = None,
) -> Dict:
    """Train with cosine decay and gradient clipping."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Cosine schedule with warmup
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    history = {"train_loss": [], "val_loss": [], "step": [], "tokens": [], "lr": []}
    step = 0
    tokens_seen = 0
    train_loss_sum = 0.0
    start_time = time.time()

    print(f"Training: max_steps={max_steps}, warmup={warmup_steps}, lr={lr}")

    for batch in train_loader:
        if step >= max_steps:
            break

        # Update LR
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(step)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        tokens_seen += input_ids.numel()

        output = model(input_ids)
        logits = output["logits"] if isinstance(output, dict) else output
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=50256,  # GPT-2 pad/eos token
        )

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if device.type == "mps":
            torch.mps.synchronize()

        train_loss_sum += loss.item()

        if step % log_every == 0 and step > 0:
            if device.type == "mps":
                torch.mps.synchronize()
            avg_loss = train_loss_sum / log_every
            elapsed = time.time() - start_time
            tok_per_sec = tokens_seen / elapsed
            print(
                f"Step {step:6d} | loss={avg_loss:.4f} | "
                f"lr={get_lr(step):.2e} | tok/s={tok_per_sec:,.0f} | "
                f"elapsed={elapsed/60:.1f}m"
            )
            train_loss_sum = 0.0

        if step % eval_every == 0 and step > 0:
            if device.type == "mps":
                torch.mps.synchronize()
            val_loss = evaluate(model, val_loader, device)
            if device.type == "mps":
                torch.mps.synchronize()
            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["step"].append(step)
            history["tokens"].append(tokens_seen)
            history["lr"].append(get_lr(step))
            print(f"  >> Validation: loss={val_loss:.4f} | ppl={math.exp(val_loss):.2f}")
            model.train()

        step += 1

    # Final eval
    val_loss = evaluate(model, val_loader, device)
    history["train_loss"].append(train_loss_sum / max(1, log_every))
    history["val_loss"].append(val_loss)
    history["step"].append(step)
    history["tokens"].append(tokens_seen)
    history["lr"].append(get_lr(step))

    print(f"\nTraining complete: {step} steps, {tokens_seen/1e6:.1f}M tokens")
    print(f"Final val loss={val_loss:.4f} | ppl={math.exp(val_loss):.2f}")

    if save_path:
        print(f"Saving checkpoint to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model.config if hasattr(model, "config") else None,
            "history": history,
            "tokens_seen": tokens_seen,
            "step": step,
        }, save_path)

    return history


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 100) -> float:
    """Evaluate on validation set."""
    model.eval()
    total_batches = 0
    # Accumulate on GPU to avoid repeated MPS sync overhead/hang
    total_loss = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in loader:
            if total_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids)
            logits = output["logits"] if isinstance(output, dict) else output
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=50256,
            )
            total_loss += loss
            total_batches += 1

    return (total_loss / max(1, total_batches)).item()


# =============================================================================
# 5. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Web-scale training on Fineweb-edu")
    parser.add_argument("--condition", required=True, help="Model condition (e.g., standard_alibi, phase_dynamic_qk_film)")
    parser.add_argument("--scale", default="20M", choices=list(SCALES.keys()))
    parser.add_argument("--max_tokens", type=int, default=100_000_000, help="Max training tokens")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/webscale"))
    parser.add_argument("--save_name", default=None)
    parser.add_argument("--subset", default="sample-10BT", help="Fineweb subset")
    parser.add_argument("--prefetch_buffer", type=int, default=None, help="Override number of HF examples to prefetch.")
    parser.add_argument("--hf_home", type=Path, default=Path("data/hf_home"), help="Repo-scoped Hugging Face home/cache root.")
    parser.add_argument("--hf_datasets_cache", type=Path, default=Path("data/hf_datasets"), help="Repo-scoped datasets cache.")
    parser.add_argument("--streaming", action="store_true", help="Stream from Hugging Face instead of using the repo-local prepared cache.")
    args = parser.parse_args()

    args.hf_home = args.hf_home.expanduser().resolve()
    args.hf_datasets_cache = args.hf_datasets_cache.expanduser().resolve()
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ["HF_HUB_CACHE"] = str(args.hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(args.hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(args.hf_datasets_cache)
    args.hf_home.mkdir(parents=True, exist_ok=True)
    args.hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = GPT2TokenizerWrapper(max_length=args.seq_len, cache_dir=os.environ["HF_HUB_CACHE"])

    # Model
    model, config = create_model(args.condition, args.scale, vocab_size=tokenizer.vocab_size, device=device)
    print(f"Model: {args.condition} at {args.scale} scale")

    # Data
    max_steps = args.max_tokens // (args.batch_size * args.seq_len)
    print(f"Max steps: {max_steps}")

    # Auto-size prefetch buffer: assume ~1000 tokens per HF example
    prefetch = args.prefetch_buffer or max(50000, args.max_tokens // 1000)
    val_prefetch = args.prefetch_buffer or max(50000, 1_000_000 // 1000)
    print(f"Prefetch buffer: {prefetch:,} examples")

    train_dataset = FinewebStreamingDataset(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_tokens=args.max_tokens,
        subset=args.subset,
        prefetch_buffer=prefetch,
        cache_dir=str(args.hf_datasets_cache),
        streaming=args.streaming,
    )
    # Small validation set from first N examples
    val_dataset = FinewebStreamingDataset(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_tokens=1_000_000,
        subset=args.subset,
        prefetch_buffer=val_prefetch,
        cache_dir=str(args.hf_datasets_cache),
        streaming=args.streaming,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Save path
    save_name = args.save_name or f"{args.condition}_{args.scale}_s{args.seed}.pt"
    save_path = args.output_dir / save_name

    # Train
    history = train_webscale(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_steps=max_steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        lr=args.lr,
        grad_clip=args.grad_clip,
        save_path=save_path,
    )

    # Save history
    history_path = args.output_dir / f"{save_name.replace('.pt', '_history.json')}"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
