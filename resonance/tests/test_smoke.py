"""Comprehensive smoke test for the Resonance Transformer infrastructure.

Runs lightweight forward passes through all swappable components to catch
integration regressions quickly.
"""

import torch

from resonance.kernels import build_kernel
from resonance.phase_embeddings import build_phase_embedding
from resonance.bias_modes import build_bias_mode
from resonance.config import ResonanceConfig
from resonance.layer_config import LayerConfig, LayerConfigRegistry


def test_kernel_phase_combinations() -> None:
    """All kernels must work with all compatible phase embeddings."""
    config = ResonanceConfig()
    B, L = 2, 8
    vocab = 100

    kernels = ["cosine", "cosine_weighted", "dot", "rbf", "laplace", "bilinear",
               "complex_magnitude", "complex_real", "attention"]
    phases = ["real", "fourier_fixed", "complex_angle", "hierarchical", "factorized"]

    for p in phases:
        kwargs: dict = {}
        if p == "hierarchical":
            kwargs["scales"] = 4
        elif p == "factorized":
            kwargs["rank"] = 8
        pe = build_phase_embedding(p, vocab, config.n_frequencies, **kwargs)
        ids = torch.randint(0, vocab, (B, L))
        phases_out = pe(ids)
        for k in kernels:
            if k in ("complex_magnitude", "complex_real") and p != "complex_angle":
                continue
            k_kwargs: dict = {}
            if k in ("rbf", "laplace"):
                k_kwargs["gamma"] = 1.0
            elif k == "bilinear":
                k_kwargs["rank"] = 8
            elif k == "attention":
                k_kwargs["temperature"] = 1.0
            kernel = build_kernel(k, config.n_frequencies, **k_kwargs)
            R = kernel(phases_out)
            assert R.shape == (B, L, L), f"{k} x {p} got {R.shape}"


def test_bias_modes() -> None:
    """All bias modes must accept [B, H, S, S] resonance and QK logits."""
    B, H, L = 2, 4, 8
    biases = ["additive", "multiplicative_gate", "residual_gate",
              "temperature_scaled", "softmax_reweighted", "only_resonance"]
    QK = torch.randn(B, H, L, L)
    R = torch.randn(B, H, L, L)
    for b in biases:
        bm = build_bias_mode(b, n_heads=H)
        out = bm(QK, R, None)
        assert out.shape == QK.shape, f"{b} shape mismatch"


def test_layer_config_registry() -> None:
    """Layer pattern matching must resolve correctly."""
    reg = LayerConfigRegistry({
        "first:2": LayerConfig(kernel="rbf", bias_mode="residual_gate", preset="strong"),
        "range:2:4": LayerConfig(kernel="complex_magnitude", bias_mode="multiplicative_gate"),
        "last:1": LayerConfig(kernel="laplace", bias_mode="only_resonance"),
    })
    assert reg.get_config(0, 6).kernel == "rbf"
    assert reg.get_config(1, 6).kernel == "rbf"
    assert reg.get_config(2, 6).kernel == "complex_magnitude"
    assert reg.get_config(5, 6).kernel == "laplace"
    assert reg.get_config(3, 6).preset is None


def test_modern_llama_forward() -> None:
    """Resonant LLaMA must produce logits of the right shape."""
    from resonance.modern.llama import LlamaConfig, ResonantLlama
    cfg = LlamaConfig(
        vocab_size=128, embed_dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        use_resonance=True, n_frequencies=16, phase_embedding="complex_angle",
        resonance_kernel="complex_magnitude", bias_mode="residual_gate",
        init_preset="strong",
    )
    model = ResonantLlama(cfg)
    x = torch.randint(0, 128, (2, 8))
    out = model(x)
    assert out["logits"].shape == (2, 8, 128)


def test_modern_qwen_forward() -> None:
    """Resonant Qwen must produce logits of the right shape."""
    from resonance.modern.qwen3_5 import Qwen3_5Config, ResonantQwen3_5
    cfg = Qwen3_5Config(
        vocab_size=128, embed_dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        use_resonance=True, n_frequencies=16, phase_embedding="complex_angle",
        resonance_kernel="complex_magnitude", bias_mode="residual_gate",
        init_preset="strong",
    )
    model = ResonantQwen3_5(cfg)
    x = torch.randint(0, 128, (2, 8))
    out = model(x)
    assert out["logits"].shape == (2, 8, 128)


def test_modern_gemma_forward() -> None:
    """Resonant Gemma4 must produce logits of the right shape."""
    from resonance.modern.gemma4 import Gemma4Config, ResonantGemma4
    cfg = Gemma4Config(
        vocab_size=128, embed_dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        use_resonance=True, n_frequencies=16, phase_embedding="complex_angle",
        resonance_kernel="complex_magnitude", bias_mode="residual_gate",
        init_preset="strong",
    )
    model = ResonantGemma4(cfg)
    x = torch.randint(0, 128, (2, 8))
    out = model(x)
    assert out["logits"].shape == (2, 8, 128)


def test_sae() -> None:
    """Sparse autoencoder training step must return positive loss."""
    from resonance.sae import SparseAutoencoder, SAETrainer
    sae = SparseAutoencoder(input_dim=64, hidden_dim=256)
    trainer = SAETrainer(sae, lr=1e-3, l1_lambda=1e-3)
    stream = torch.randn(100, 64)
    metrics = trainer.train_step(stream)
    assert metrics["loss"] > 0


def test_probe() -> None:
    """Linear probe training must complete and return metrics."""
    from resonance.probes import LinearProbe, ProbeTrainer
    probe = LinearProbe(input_dim=64, output_dim=10, task="classification")
    trainer = ProbeTrainer(probe, lr=1e-3)
    X_train, X_val = torch.randn(40, 64), torch.randn(10, 64)
    y_train, y_val = torch.randint(0, 10, (40,)), torch.randint(0, 10, (10,))
    result = trainer.train(X_train, y_train, x_val=X_val, y_val=y_val,
                           epochs=2, batch_size=10)
    assert "accuracy" in result or "r2" in result


def test_init_presets() -> None:
    """All documented presets must be resolvable."""
    from resonance.modern.common import get_preset_values
    for p in ["default", "wide", "strong", "very_strong", "normalized"]:
        vals = get_preset_values(p)
        assert len(vals) == 2


def test_hard_negatives() -> None:
    """Graph-to-story hard negatives must have no collisions and valid mutations."""
    from synthetic.semantic_story import verify_hard_negatives
    stats = verify_hard_negatives(50, seed=42)
    assert stats["positive_negative_collisions"] == 0
    assert stats["valid_single_mutation"] == stats["total_negatives"]


def test_interpretability() -> None:
    """Trace and intervention context managers must run without error."""
    from resonance.interpretability import (
        trace_resonance_forward,
        phase_intervention,
        resonance_weight_scale,
        phase_rows_patch,
        phase_rows_copy,
    )
    from resonance.models import ResonanceTransformer

    config = ResonanceConfig(vocab_size=100, embed_dim=64, n_layers=2, n_heads=4, n_frequencies=16)
    model = ResonanceTransformer(config)
    x = torch.randint(0, 100, (2, 8))

    result = trace_resonance_forward(model, x)
    assert result.resonance_matrix.shape == (2, 8, 8)

    with phase_intervention(model, mode="noise", sigma=0.1):
        model(x)
    with resonance_weight_scale(model, 2.0):
        model(x)
    with phase_rows_patch(model, [0, 1], torch.randn(2, 16)):
        model(x)
    with phase_rows_copy(model, source_token_ids=[0], target_token_ids=[1]):
        model(x)


def test_per_layer_config() -> None:
    """Per-layer kernel/bias overrides must be applied correctly."""
    from resonance.models import ResonanceTransformer

    config = ResonanceConfig(vocab_size=100, embed_dim=64, n_layers=4, n_heads=4, n_frequencies=16)
    reg = LayerConfigRegistry({
        "first:2": LayerConfig(kernel="rbf", bias_mode="residual_gate", preset="strong"),
        "range:2:4": LayerConfig(kernel="complex_magnitude", bias_mode="multiplicative_gate"),
    })
    model = ResonanceTransformer(config, layer_configs=reg)
    x = torch.randint(0, 100, (2, 8))
    out = model(x)
    logits = out["logits"] if isinstance(out, dict) else out
    assert logits.shape == (2, 8, 100)
    assert model.blocks[0].kernel is not None
    assert model.blocks[2].kernel is not None


if __name__ == "__main__":
    test_kernel_phase_combinations()
    test_bias_modes()
    test_layer_config_registry()
    test_modern_llama_forward()
    test_modern_qwen_forward()
    test_modern_gemma_forward()
    test_sae()
    test_probe()
    test_init_presets()
    test_hard_negatives()
    test_interpretability()
    test_per_layer_config()
    print("=== ALL SMOKE TESTS PASSED ===")
