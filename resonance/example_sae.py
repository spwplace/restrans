"""Minimal example: train a Sparse Autoencoder on a ResonanceTransformer stream."""

from __future__ import annotations

import torch

from resonance import ResonanceConfig, ResonanceTransformer, set_seed
from resonance.sae import (
    SAETrainer,
    SparseAutoencoder,
    StreamExtractor,
    evaluate_sae,
)


def main() -> None:
    set_seed(42)

    # ------------------------------------------------------------------
    # 1. Tiny ResonanceTransformer
    # ------------------------------------------------------------------
    config = ResonanceConfig(
        name="tiny_sae_demo",
        vocab_size=256,
        max_seq_len=64,
        embed_dim=128,
        n_layers=2,
        n_heads=4,
        ff_dim=512,
        n_frequencies=16,
        dropout=0.0,
        use_phase_stream=True,
        use_resonance_bias=True,
    )
    model = ResonanceTransformer(config)
    model.eval()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # 2. Synthetic data
    # ------------------------------------------------------------------
    batch_size = 32
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # ------------------------------------------------------------------
    # 3. Extract streams
    # ------------------------------------------------------------------
    extractor = StreamExtractor(model)
    streams = extractor.extract(input_ids)
    print(f"\nExtracted {len(streams)} streams:")
    for name, tensor in streams.items():
        print(f"  {name}: {tuple(tensor.shape)}")

    # ------------------------------------------------------------------
    # 4. Train an SAE on the final hidden states
    # ------------------------------------------------------------------
    stream_name = "final_hidden"
    activations = streams[stream_name]  # (batch, seq_len, embed_dim)
    # Flatten to (batch*seq_len, embed_dim) for the SAE
    flat_activations = activations.view(-1, activations.shape[-1])

    sae = SparseAutoencoder(
        input_dim=config.embed_dim,
        hidden_dim=4 * config.embed_dim,
    )
    trainer = SAETrainer(sae, lr=1e-3, l1_lambda=1e-3)

    n_epochs = 5
    print(f"\nTraining SAE on '{stream_name}' ({flat_activations.shape[0]} samples) ...")
    for epoch in range(n_epochs):
        metrics = trainer.train_epoch(flat_activations, batch_size=256, shuffle=True)
        print(
            f"  Epoch {epoch + 1}/{n_epochs} | "
            f"loss={metrics['loss']:.4f} | "
            f"recon_mse={metrics['recon_mse']:.4f} | "
            f"sparsity={metrics['sparsity_penalty']:.4f} | "
            f"dead={metrics['dead_features']:.2%}"
        )

    # ------------------------------------------------------------------
    # 5. Evaluation metrics
    # ------------------------------------------------------------------
    eval_metrics = evaluate_sae(sae, flat_activations)
    print("\nFinal SAE metrics:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Quick sanity-check: save + load checkpoint
    trainer.save_checkpoint("outputs/sae_demo.pt")
    trainer.load_checkpoint("outputs/sae_demo.pt")
    print("\nCheckpoint save/load OK.")


if __name__ == "__main__":
    main()
