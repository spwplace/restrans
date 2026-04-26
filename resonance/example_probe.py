"""Minimal example: train linear probes on semantic vs. phase streams.

This script creates a tiny ResonanceTransformer, generates synthetic token
sequences with a simple label (even / odd token ID), extracts semantic and
phase representations, trains linear probes on both streams, and compares
their accuracy.
"""

from __future__ import annotations

import torch

from resonance import ResonanceConfig, ResonanceTransformer, set_seed
from resonance.probes import (
    LinearProbe,
    ProbeTrainer,
    RepresentationExtractor,
    compare_probes,
)


def main() -> None:
    set_seed(42)

    # ------------------------------------------------------------------
    # 1. Tiny ResonanceTransformer
    # ------------------------------------------------------------------
    config = ResonanceConfig(
        name="tiny_probe_demo",
        vocab_size=128,
        max_seq_len=64,
        embed_dim=64,
        n_layers=2,
        n_heads=4,
        ff_dim=256,
        n_frequencies=16,
        dropout=0.0,
        use_phase_stream=True,
        use_resonance_bias=True,
    )
    model = ResonanceTransformer(config)
    model.eval()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # 2. Synthetic data with known labels
    # ------------------------------------------------------------------
    batch_size = 256
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Label: even / odd based on the *first* token ID
    labels = (input_ids[:, 0] % 2).long()  # 0 = even, 1 = odd

    # Split into train / val
    n_train = int(0.8 * batch_size)
    train_ids, val_ids = input_ids[:n_train], input_ids[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]

    # ------------------------------------------------------------------
    # 3. Extract representations
    # ------------------------------------------------------------------
    extractor = RepresentationExtractor(model)

    # Semantic embeddings
    semantic_streams = extractor.extract_and_aggregate(
        train_ids,
        streams=["semantic_embed"],
        aggregation="first",
    )
    semantic_train = semantic_streams["semantic_embed"]

    semantic_streams_val = extractor.extract_and_aggregate(
        val_ids,
        streams=["semantic_embed"],
        aggregation="first",
    )
    semantic_val = semantic_streams_val["semantic_embed"]

    # Phase projections
    phase_streams = extractor.extract_and_aggregate(
        train_ids,
        streams=["phase_proj"],
        aggregation="first",
    )
    phase_train = phase_streams["phase_proj"]

    phase_streams_val = extractor.extract_and_aggregate(
        val_ids,
        streams=["phase_proj"],
        aggregation="first",
    )
    phase_val = phase_streams_val["phase_proj"]

    print(f"\nSemantic train shape: {tuple(semantic_train.shape)}")
    print(f"Phase train shape:    {tuple(phase_train.shape)}")

    # ------------------------------------------------------------------
    # 4. Train probes
    # ------------------------------------------------------------------
    n_classes = 2

    # Semantic probe
    probe_sem = LinearProbe(config.embed_dim, n_classes, task="classification")
    trainer_sem = ProbeTrainer(probe_sem, lr=1e-2)
    print("\nTraining semantic probe …")
    trainer_sem.train(
        semantic_train,
        train_labels,
        semantic_val,
        val_labels,
        epochs=50,
        batch_size=64,
        patience=5,
        verbose=True,
    )
    sem_results = trainer_sem.evaluate(semantic_val, val_labels)

    # Phase probe
    probe_phase = LinearProbe(config.embed_dim, n_classes, task="classification")
    trainer_phase = ProbeTrainer(probe_phase, lr=1e-2)
    print("\nTraining phase probe …")
    trainer_phase.train(
        phase_train,
        train_labels,
        phase_val,
        val_labels,
        epochs=50,
        batch_size=64,
        patience=5,
        verbose=True,
    )
    phase_results = trainer_phase.evaluate(phase_val, val_labels)

    # ------------------------------------------------------------------
    # 5. Compare
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Probe comparison (even/odd classification)")
    print("=" * 50)
    compare_probes({
        "semantic_embed": sem_results,
        "phase_proj": phase_results,
    })

    # ------------------------------------------------------------------
    # 6. Quick regression demo (predict token ID magnitude)
    # ------------------------------------------------------------------
    reg_labels = input_ids[:, 0].float() / config.vocab_size  # normalised
    train_reg = reg_labels[:n_train]
    val_reg = reg_labels[n_train:]

    probe_reg = LinearProbe(config.embed_dim, 1, task="regression")
    trainer_reg = ProbeTrainer(probe_reg, lr=1e-2)
    print("\nTraining regression probe on semantic stream …")
    trainer_reg.train(
        semantic_train,
        train_reg,
        semantic_val,
        val_reg,
        epochs=50,
        batch_size=64,
        patience=5,
        verbose=True,
    )
    reg_results = trainer_reg.evaluate(semantic_val, val_reg)
    print(f"\nRegression R²: {reg_results['r2']:.4f}")


if __name__ == "__main__":
    main()
