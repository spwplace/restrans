"""Resonance transformer package.

A novel transformer architecture that augments standard semantic embeddings
with phase (structural/phonetic) embeddings and resonance-biased attention.
"""

from .config import ResonanceConfig, StandardConfig, count_params
from .data import TextDataset
from .models import (
    ResonanceAttention,
    ResonanceBlock,
    ResonanceEmbedding,
    ResonanceTransformer,
    StandardAttention,
    StandardBlock,
    StandardTransformer,
)
from .phonetic import build_rhyme_index
from .training import train_model
from .utils import count_module_params, generate_synthetic_text, set_seed

__all__ = [
    "StandardConfig",
    "ResonanceConfig",
    "count_params",
    "count_module_params",
    "generate_synthetic_text",
    "set_seed",
    "StandardAttention",
    "StandardBlock",
    "StandardTransformer",
    "ResonanceEmbedding",
    "ResonanceAttention",
    "ResonanceBlock",
    "ResonanceTransformer",
    "build_rhyme_index",
    "TextDataset",
    "train_model",
]
