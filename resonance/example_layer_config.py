"""Demonstration of per-layer kernel/bias configuration for the Resonance Transformer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from resonance.config import ResonanceConfig
from resonance.layer_config import LayerConfig, LayerConfigRegistry
from resonance.models import ResonanceTransformer


def main() -> None:
    """Build a ResonanceTransformer with per-layer configs and verify them."""
    # Small config for demo
    config = ResonanceConfig(
        vocab_size=1000,
        max_seq_len=64,
        embed_dim=128,
        n_layers=4,
        n_heads=4,
        ff_dim=512,
        n_frequencies=16,
        resonance_kernel="cosine",
        bias_mode="additive",
    )

    # Per-layer configuration:
    #   Layer 0-1: cosine kernel + additive bias (local structure)
    #   Layer 2-3: complex_magnitude kernel + residual_gate bias (global coherence)
    registry = LayerConfigRegistry({
        "first:2": LayerConfig(kernel="cosine", bias_mode="additive"),
        "last:2": LayerConfig(kernel="complex_magnitude", bias_mode="residual_gate"),
    })

    model = ResonanceTransformer(config, layer_configs=registry)

    # Verify each layer has the expected kernel and bias_mode
    expected = [
        ("CosineKernel", "AdditiveBias"),
        ("CosineKernel", "AdditiveBias"),
        ("ComplexMagnitudeKernel", "ResidualGateBias"),
        ("ComplexMagnitudeKernel", "ResidualGateBias"),
    ]

    for i, block in enumerate(model.blocks):
        kernel_cls = block.kernel.__class__.__name__ if block.kernel is not None else "None"
        bias_cls = block.attn.bias_mode.__class__.__name__
        exp_kernel, exp_bias = expected[i]

        assert kernel_cls == exp_kernel, (
            f"Layer {i}: expected kernel {exp_kernel}, got {kernel_cls}"
        )
        assert bias_cls == exp_bias, (
            f"Layer {i}: expected bias_mode {exp_bias}, got {bias_cls}"
        )
        print(f"Layer {i}: kernel={kernel_cls}, bias_mode={bias_cls} ✓")

    # Forward pass
    import torch

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    print(f"\nForward pass output shape: {output['logits'].shape}")
    print("All assertions passed!")


if __name__ == "__main__":
    main()
