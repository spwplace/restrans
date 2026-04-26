"""Example: Modern architectures with init_preset and complex phase integration."""

from __future__ import annotations

import torch

from resonance.modern.llama import LlamaConfig, ResonantLlama


def main() -> None:
    device = torch.device("cpu")

    # --- 1. Strong preset: larger phase init std and resonance weight ---
    config_strong = LlamaConfig(
        vocab_size=128,
        max_seq_len=64,
        embed_dim=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ff_dim=512,
        use_resonance=True,
        n_frequencies=16,
        init_preset="strong",
    )
    model_strong = ResonantLlama(config_strong).to(device)

    # Verify strong preset gives larger values
    phase_std_strong = model_strong.embed.phase.weight.std().item()
    attn_weight_strong = model_strong.layers[0].attn.resonance_weight.mean().item()
    print(f"[strong] phase init std: {phase_std_strong:.3f}")
    print(f"[strong] resonance weight: {attn_weight_strong:.3f}")

    # --- 2. Default preset: smaller values ---
    config_default = LlamaConfig(
        vocab_size=128,
        max_seq_len=64,
        embed_dim=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ff_dim=512,
        use_resonance=True,
        n_frequencies=16,
        init_preset="default",
    )
    model_default = ResonantLlama(config_default).to(device)

    phase_std_default = model_default.embed.phase.weight.std().item()
    attn_weight_default = model_default.layers[0].attn.resonance_weight.mean().item()
    print(f"[default] phase init std: {phase_std_default:.3f}")
    print(f"[default] resonance weight: {attn_weight_default:.3f}")

    assert phase_std_strong > phase_std_default, "strong preset should have larger phase std"
    assert attn_weight_strong > attn_weight_default, "strong preset should have larger resonance weight"
    print("✓ Preset verification passed")

    # --- 3. Complex angle phase + complex magnitude kernel ---
    config_complex = LlamaConfig(
        vocab_size=128,
        max_seq_len=64,
        embed_dim=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ff_dim=512,
        use_resonance=True,
        n_frequencies=16,
        phase_embedding="complex_angle",
        resonance_kernel="complex_magnitude",
        init_preset="strong",
    )
    model_complex = ResonantLlama(config_complex).to(device)

    # Forward pass
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config_complex.vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        output = model_complex(input_ids)

    logits = output["logits"]
    print(f"[complex] logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config_complex.vocab_size)
    print("✓ Complex phase forward pass passed")

    # Verify that complex path was used for resonance matrix
    resonance = model_complex.embed.get_resonance_matrix(input_ids)
    print(f"[complex] resonance shape: {resonance.shape}")
    assert resonance.shape == (batch_size, seq_len, seq_len)
    assert not torch.is_complex(resonance), "resonance output should be real"
    print("✓ Complex resonance matrix passed")


if __name__ == "__main__":
    main()
