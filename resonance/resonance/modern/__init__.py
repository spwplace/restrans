"""Modern transformer architectures: Llama, Qwen3.5, Gemma4 (dense + MoE + resonance variants)."""

from __future__ import annotations

from .llama import LlamaConfig, LlamaTransformer, LlamaMoE, ResonantLlama, ResonantLlamaMoE
from .qwen3_5 import Qwen3_5Config, Qwen3_5Transformer, Qwen3_5MoE, ResonantQwen3_5, ResonantQwen3_5MoE
from .gemma4 import Gemma4Config, Gemma4Transformer, Gemma4MoE, ResonantGemma4, ResonantGemma4MoE

__all__ = [
    "LlamaConfig",
    "LlamaTransformer",
    "LlamaMoE",
    "ResonantLlama",
    "ResonantLlamaMoE",
    "Qwen3_5Config",
    "Qwen3_5Transformer",
    "Qwen3_5MoE",
    "ResonantQwen3_5",
    "ResonantQwen3_5MoE",
    "Gemma4Config",
    "Gemma4Transformer",
    "Gemma4MoE",
    "ResonantGemma4",
    "ResonantGemma4MoE",
]
