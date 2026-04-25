"""Training loop for Standard and Resonance transformers."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import ResonanceConfig, StandardConfig


def train_model(
    model: nn.Module,
    config: StandardConfig | ResonanceConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    n_epochs: int = 20,
    device: torch.device | str | None = None,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    log_interval: int = 500,
) -> dict[str, list[float]]:
    """Train a language model and return per-epoch metrics.

    The training loop performs gradient accumulation as configured by
    ``config.gradient_accumulation``, clips gradients by global norm,
    and logs validation perplexity at the end of every epoch.

    Args:
        model: A ``StandardTransformer`` or ``ResonanceTransformer``.
        config: Configuration dataclass (must expose ``batch_size``,
            ``gradient_accumulation``, and ``learning_rate``).
        train_dataset: Training dataset yielding ``{"input_ids", "labels"}``.
        val_dataset: Validation dataset with the same structure.
        n_epochs: Number of passes over the training data.
        device: Torch device (or string).  If ``None``, CUDA is used when
            available, otherwise CPU.
        max_grad_norm: Maximum gradient norm for clipping.
        weight_decay: AdamW weight decay.
        log_interval: Print batch-level loss every *N* batches.

    Returns:
        Dictionary mapping ``"train_loss"``, ``"val_loss"``, and
        ``"val_ppl"`` to lists of per-epoch values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=weight_decay,
    )

    print(f"\nTraining: {config.name}")
    print(f"  Device: {device}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Gradient accumulation: {config.gradient_accumulation}")

    results: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
    }

    for epoch in range(n_epochs):
        # -----------------------------------------------------------------
        # Training
        # -----------------------------------------------------------------
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, labels=labels)["loss"]
            # Scale loss for gradient accumulation
            (loss / config.gradient_accumulation).backward()

            if (i + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

            if (i + 1) % log_interval == 0:
                print(
                    f"    Batch {i + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Step any remaining accumulated gradients
        if (len(train_loader) % config.gradient_accumulation) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        epoch_time = time.time() - start_time
        avg_train = train_loss / len(train_loader)

        # -----------------------------------------------------------------
        # Validation
        # -----------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                val_loss += model(input_ids, labels=labels)["loss"].item()

        avg_val = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val)

        results["train_loss"].append(avg_train)
        results["val_loss"].append(avg_val)
        results["val_ppl"].append(val_ppl)

        # Log embedding blend if available (Resonance models)
        blend_str = ""
        if hasattr(model, "embedding") and hasattr(model.embedding, "blend"):
            blend = torch.sigmoid(model.embedding.blend).mean().item()
            blend_str = f"\n    Embedding blend: {blend:.3f}"

        print(
            f"\n  Epoch {epoch + 1}/{n_epochs} ({epoch_time / 60:.1f} min)"
        )
        print(f"    Train Loss: {avg_train:.4f}")
        print(f"    Val Loss:   {avg_val:.4f}")
        print(f"    Val PPL:    {val_ppl:.2f}{blend_str}")

    return results
