#!/usr/bin/env python3
"""Baseline training script for Standard and Resonance transformers.

Trains three tiny models on synthetic data for a single epoch and prints
validation perplexities.  Designed to run offline without downloading
large datasets or model weights from the internet.

Usage::

    python scripts/baseline_train.py

The script automatically falls back to a minimal tokenizer if the GPT-2
tokenizer cannot be downloaded, and always uses synthetic text data so no
``datasets`` library is required.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Ensure the package is importable when running from the repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from resonance import (
    ResonanceConfig,
    ResonanceTransformer,
    StandardConfig,
    StandardTransformer,
    TextDataset,
    generate_synthetic_text,
    set_seed,
    train_model,
)

# ---------------------------------------------------------------------------
# Tiny configs for fast baseline experiments
# ---------------------------------------------------------------------------

TINY_STANDARD = StandardConfig(
    name="tiny_standard",
    vocab_size=500,
    max_seq_len=32,
    embed_dim=64,
    n_layers=2,
    n_heads=4,
    ff_dim=256,
    dropout=0.1,
    batch_size=4,
    gradient_accumulation=2,
    learning_rate=3e-4,
)

TINY_RESONANCE_RANDOM = ResonanceConfig(
    name="tiny_resonance_random",
    vocab_size=500,
    max_seq_len=32,
    embed_dim=64,
    n_layers=2,
    n_heads=4,
    ff_dim=256,
    n_frequencies=16,
    resonance_blend=0.3,
    resonance_attn_weight=0.1,
    dropout=0.1,
    batch_size=4,
    gradient_accumulation=2,
    learning_rate=3e-4,
    phonetic_init=False,
)

TINY_RESONANCE_PHONETIC = ResonanceConfig(
    name="tiny_resonance_phonetic",
    vocab_size=500,
    max_seq_len=32,
    embed_dim=64,
    n_layers=2,
    n_heads=4,
    ff_dim=256,
    n_frequencies=16,
    resonance_blend=0.3,
    resonance_attn_weight=0.1,
    dropout=0.1,
    batch_size=4,
    gradient_accumulation=2,
    learning_rate=3e-4,
    phonetic_init=True,
)

CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_EPOCHS = 1
MAX_TRAIN_EXAMPLES = 1_000
MAX_VAL_EXAMPLES = 200


class _MinimalTokenizer:
    """A tiny word-hash tokenizer that works offline.

    Splits text on whitespace and hashes each word into the range
    ``[0, vocab_size)``.  This is not a *good* tokenizer, but it is
    sufficient for verifying that the training pipeline works end-to-end
    without downloading anything from the internet.
    """

    def __init__(self, vocab_size: int = 500) -> None:
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        # Reserve id 0 for pad
        self._pad_id = 0

    def encode(self, text: str, truncation: bool = False) -> list[int]:
        words = text.split()
        ids = [(hash(w) % (self.vocab_size - 1)) + 1 for w in words]
        return ids

    def decode(self, token_ids: list[int] | int) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(f"tok{i}" for i in token_ids)

    def __len__(self) -> int:
        return self.vocab_size


def _get_tokenizer(vocab_size: int):
    """Load GPT-2 tokenizer if possible, otherwise use minimal fallback."""
    try:
        from transformers import GPT2Tokenizer

        # local_files_only=True avoids hanging on network failure
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)

        # Sanity check: the cached tokenizer might be corrupted
        test_ids = tokenizer.encode("hello world")
        if len(test_ids) == 0 or len(tokenizer) < 100:
            raise RuntimeError(
                f"Cached tokenizer appears corrupted (vocab={len(tokenizer)}, "
                f"test_encode={test_ids})."
            )

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        print("Using GPT-2 tokenizer (cached).")
        return tokenizer
    except Exception as exc:
        print(f"GPT-2 tokenizer unavailable ({exc}).")
        print(f"Falling back to minimal word-hash tokenizer (vocab={vocab_size}).")
        return _MinimalTokenizer(vocab_size)


def _build_rhyme_index(tokenizer, vocab_size: int):
    """Build rhyme index for phonetic initialisation if possible."""
    try:
        from resonance.phonetic import build_rhyme_index

        return build_rhyme_index(tokenizer, vocab_size)
    except Exception as exc:
        print(f"Warning: could not build rhyme index ({exc}).")
        return None


def main() -> int:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    tokenizer = _get_tokenizer(TINY_STANDARD.vocab_size)
    vocab_size = len(tokenizer)

    # Override vocab sizes
    TINY_STANDARD.vocab_size = vocab_size
    TINY_RESONANCE_RANDOM.vocab_size = vocab_size
    TINY_RESONANCE_PHONETIC.vocab_size = vocab_size

    # Generate synthetic data
    print("\nGenerating synthetic dataset...")
    all_docs = generate_synthetic_text(
        n_documents=3_000, min_words=50, max_words=300, seed=42
    )
    train_docs = all_docs[:2_500]
    val_docs = all_docs[2_500:]
    print(f"  {len(train_docs)} train documents, {len(val_docs)} val documents")

    print("Building training dataset...")
    train_dataset = TextDataset(
        train_docs,
        tokenizer,
        max_length=TINY_STANDARD.max_seq_len,
        max_examples=MAX_TRAIN_EXAMPLES,
    )

    print("Building validation dataset...")
    val_dataset = TextDataset(
        val_docs,
        tokenizer,
        max_length=TINY_STANDARD.max_seq_len,
        max_examples=MAX_VAL_EXAMPLES,
    )

    # Build rhyme index once
    rhyme_index = _build_rhyme_index(tokenizer, vocab_size)
    if rhyme_index:
        print(
            f"Rhyme index: {rhyme_index['tokens_with_rhyme']} / {vocab_size} "
            f"tokens in {rhyme_index['n_rhyme_groups']} groups."
        )

    all_results: dict[str, dict[str, list[float]]] = {}

    # -----------------------------------------------------------------
    # 1. Standard Transformer (baseline)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Standard Transformer (baseline)")
    print("=" * 70)
    model_std = StandardTransformer(TINY_STANDARD)
    print(f"  Parameters: {model_std.n_params / 1e6:.2f}M")

    results_std = train_model(
        model_std,
        TINY_STANDARD,
        train_dataset,
        val_dataset,
        n_epochs=N_EPOCHS,
        device=device,
        log_interval=50,
    )
    all_results["standard"] = results_std

    ckpt_std = CHECKPOINT_DIR / "standard.pt"
    torch.save(model_std.state_dict(), ckpt_std)
    print(f"  Checkpoint saved to {ckpt_std}")

    del model_std
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # 2. Resonance Transformer (random phase init)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Resonance Transformer (random phase init)")
    print("=" * 70)
    model_rand = ResonanceTransformer(TINY_RESONANCE_RANDOM, rhyme_index=None)
    print(f"  Parameters: {model_rand.n_params / 1e6:.2f}M")

    results_rand = train_model(
        model_rand,
        TINY_RESONANCE_RANDOM,
        train_dataset,
        val_dataset,
        n_epochs=N_EPOCHS,
        device=device,
        log_interval=50,
    )
    all_results["resonance_random"] = results_rand

    ckpt_rand = CHECKPOINT_DIR / "resonance_random.pt"
    torch.save(model_rand.state_dict(), ckpt_rand)
    print(f"  Checkpoint saved to {ckpt_rand}")

    del model_rand
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # 3. Resonance Transformer (phonetic phase init)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Resonance Transformer (phonetic phase init)")
    print("=" * 70)
    model_phone = ResonanceTransformer(
        TINY_RESONANCE_PHONETIC, rhyme_index=rhyme_index
    )
    print(f"  Parameters: {model_phone.n_params / 1e6:.2f}M")

    results_phone = train_model(
        model_phone,
        TINY_RESONANCE_PHONETIC,
        train_dataset,
        val_dataset,
        n_epochs=N_EPOCHS,
        device=device,
        log_interval=50,
    )
    all_results["resonance_phonetic"] = results_phone

    ckpt_phone = CHECKPOINT_DIR / "resonance_phonetic.pt"
    torch.save(model_phone.state_dict(), ckpt_phone)
    print(f"  Checkpoint saved to {ckpt_phone}")

    del model_phone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Val Loss':<12} {'Val PPL':<12}")
    print("-" * 54)

    for name, results in all_results.items():
        val_loss = results["val_loss"][-1]
        val_ppl = results["val_ppl"][-1]
        print(f"{name:<30} {val_loss:<12.4f} {val_ppl:<12.2f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
