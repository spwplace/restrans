"""Iso-parameter model builder.

Builds StandardTransformer and ResonanceTransformer with matched parameter
counts so ablation comparisons are fair.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from resonance.config import ResonanceConfig, StandardConfig
from resonance.models import ResonanceTransformer, StandardTransformer


def _count_params_at_dim(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    ff_dim: int,
    is_resonance: bool = False,
    n_frequencies: int = 32,
) -> int:
    """Count parameters for a transformer with given dims."""
    # Embeddings
    params = vocab_size * embed_dim  # token / semantic
    params += seq_len * embed_dim    # positional

    if is_resonance:
        params += vocab_size * n_frequencies  # phase
        params += n_frequencies * embed_dim    # phase_proj
        params += embed_dim                    # blend

    # Per layer: LN(2*D) + QKV(3*D*D) + out(D*D) + ff(D*ff + ff*D) + LN(2*D)
    layer_params = (
        2 * embed_dim  # ln1
        + 3 * embed_dim * embed_dim  # qkv
        + embed_dim * embed_dim      # out_proj
        + embed_dim * ff_dim + ff_dim * embed_dim  # ff
        + 2 * embed_dim  # ln2
    )
    if is_resonance:
        layer_params += n_heads  # resonance_weight

    params += n_layers * layer_params
    params += 2 * embed_dim  # ln_final
    # lm_head is tied, so 0 extra

    return params


def find_iso_param_dim(
    target_params: int,
    vocab_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    ff_dim: int,
    is_resonance: bool = False,
    n_frequencies: int = 32,
    min_dim: int = 32,
    max_dim: int = 2048,
) -> int:
    """Find embed_dim that gives closest parameter count to target."""
    best_dim = min_dim
    best_diff = float("inf")

    for dim in range(min_dim, max_dim + 1, n_heads):  # must divide n_heads
        p = _count_params_at_dim(
            vocab_size, seq_len, dim, n_layers, n_heads, ff_dim,
            is_resonance, n_frequencies,
        )
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_dim = dim

    return best_dim


def build_iso_pair(
    base_embed_dim: int = 256,
    n_layers: int = 4,
    n_heads: int = 8,
    ff_dim: int = 1024,
    vocab_size: int = 512,
    seq_len: int = 64,
    n_frequencies: int = 32,
    dropout: float = 0.1,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    gradient_accumulation: int = 1,
    phase_init_std: float = 0.3,
    resonance_attn_weight: float = 0.1,
    resonance_blend: float = 0.3,
    phonetic_init: bool = False,
    use_phase_stream: bool = True,
    use_resonance_bias: bool = True,
) -> tuple[StandardTransformer, ResonanceTransformer, StandardConfig, ResonanceConfig]:
    """Build a standard and resonance model with matched parameter counts.

    The resonance model uses the requested embed_dim. The standard model's
    embed_dim is adjusted downward so total params match as closely as
    possible. Both dimensions are rounded to multiples of n_heads.
    """
    # Build resonance config first
    res_config = ResonanceConfig(
        name="resonance_iso",
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        embed_dim=base_embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        n_frequencies=n_frequencies,
        resonance_blend=resonance_blend,
        resonance_attn_weight=resonance_attn_weight,
        dropout=dropout,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        phase_init_std=phase_init_std,
        phonetic_init=phonetic_init,
        use_phase_stream=use_phase_stream,
        use_resonance_bias=use_resonance_bias,
    )
    res_model = ResonanceTransformer(res_config)
    res_params = sum(p.numel() for p in res_model.parameters())

    # Find standard dim that matches
    std_dim = find_iso_param_dim(
        target_params=res_params,
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        is_resonance=False,
    )

    std_config = StandardConfig(
        name="standard_iso",
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        embed_dim=std_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
    )
    std_model = StandardTransformer(std_config)
    std_params = sum(p.numel() for p in std_model.parameters())

    return std_model, res_model, std_config, res_config


def print_param_table(
    models: dict[str, nn.Module],
    configs: dict[str, Any],
) -> None:
    """Print a formatted parameter count table."""
    print("\n" + "=" * 60)
    print("Parameter Count Comparison")
    print("=" * 60)
    print(f"{'Model':<20} {'Params':>12} {'Ratio':>10}")
    print("-" * 60)

    base_params = None
    for name in sorted(models.keys()):
        p = sum(param.numel() for param in models[name].parameters())
        if base_params is None:
            base_params = p
            ratio = 1.0
        else:
            ratio = p / base_params
        print(f"{name:<20} {p:>12,} {ratio:>10.4f}")
    print("=" * 60)
