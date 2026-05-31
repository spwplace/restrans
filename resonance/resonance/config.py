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
    # Baseline attention variants used for literature-grounded comparisons.
    # ``standard`` is the original learned absolute-position baseline.
    # ``alibi`` adds a causal distance prior to attention logits.
    # ``deberta_lite`` adds relative content-position and position-content
    # terms as a compact DeBERTa-style disentangled attention comparator.
    attention_variant: str = "standard"  # standard | alibi | deberta_lite


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
    phase_init_std: float = 0.3
    resonance_blend: float = 0.3
    resonance_attn_weight: float = 0.1
    dropout: float = 0.1
    batch_size: int = 8
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4
    phonetic_init: bool = False
    use_phase_stream: bool = True
    use_resonance_bias: bool = True
    center_resonance: bool = False
    normalize_resonance: bool = False
    # Kernel selection and hyperparameters
    resonance_kernel: str = "cosine"
    kernel_gamma: float = 1.0
    kernel_learnable_gamma: bool = False
    kernel_rank: int | None = None
    kernel_temperature: float = 1.0
    # Walk kernel (walkformer) hyperparameters
    walk_atoms: int = 5
    walk_use_chiral: bool = True
    walk_band: int = 8
    walk_use_phase_drive: bool = True
    # Walkformer v1/v2 atom set switch.  "v1" keeps only the five original v1
    # atoms (byte-stable v1 behaviour); "v2" enables all ten; a v2-atom name
    # (bessel|coined|twohorn|powerlaw|learnedH) enables the v1 five plus that one.
    walk_atom_set: str = "v1"
    # Phase embedding variant
    phase_embedding: str = "real"
    phase_embedding_rank: int = 8
    phase_embedding_scales: int = 4
    # Bias application mode
    bias_mode: str = "additive"
    bias_gate_init: float = 0.0
    # Initialization preset
    init_preset: str = "default"
    # Experimental structural-channel variants
    phase_update_mode: str = "none"          # none | mlp | self_attn
    phase_update_scale: float = 0.1
    phase_update_hidden_mult: int = 2
    phase_update_heads: int = 1
    phase_condition_qk: str = "none"         # none | film
    n_structural_heads: int = 0              # heads whose logits come from phase structure
    structural_head_scale: float = 1.0
    relation_value_mode: str = "none"        # none | additive
    aux_phase_loss_weight: float = 0.0
    # Optional ordinary positional/logit prior for resonance models.  This lets
    # us test whether phase/QK conditioning is only losing because it lacks the
    # strong addressing prior used by ALiBi baselines.
    attention_variant: str = "standard"      # standard | alibi


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
