"""Configuration dataclasses for Standard and Resonance transformers."""

from dataclasses import dataclass


@dataclass
class StandardConfig:
    """Configuration for the standard (vanilla) transformer baseline.

    A GPT-like causal language model with no resonance mechanisms.
    """

    name: str = "standard_baseline"
    vocab_size: int = 50257
    max_seq_len: int = 256
    embed_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ff_dim: int = 3072
    dropout: float = 0.1
    batch_size: int = 8
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4


@dataclass
class ResonanceConfig:
    """Configuration for the Resonance transformer.

    Extends the standard transformer with dual embeddings (semantic + phase)
    and resonance-biased attention.  ``phonetic_init`` controls whether phase
    embeddings are initialised from phonetic (rhyme-based) structure.
    """

    name: str = "base"
    vocab_size: int = 50257
    max_seq_len: int = 256
    embed_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ff_dim: int = 3072
    n_frequencies: int = 32
    resonance_blend: float = 0.3
    resonance_attn_weight: float = 0.1
    dropout: float = 0.1
    batch_size: int = 8
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4
    phonetic_init: bool = False
    use_phase_stream: bool = True
    use_resonance_bias: bool = True


def count_params(config: StandardConfig | ResonanceConfig) -> int:
    """Estimate parameter count from a configuration object.

    Args:
        config: A ``StandardConfig`` or ``ResonanceConfig`` instance.

    Returns:
        Estimated total parameter count (embeddings + attention + FFN).
    """
    embed = config.vocab_size * config.embed_dim
    phase = 0
    if hasattr(config, "n_frequencies"):
        phase = config.vocab_size * config.n_frequencies
    # Q, K, V, O projections: 4 * embed_dim**2 per layer
    attn = config.n_layers * (4 * config.embed_dim**2)
    # Two linear layers: embed_dim -> ff_dim -> embed_dim
    ff = config.n_layers * (2 * config.embed_dim * config.ff_dim)
    return embed + phase + attn + ff
