"""Modern transformer architectures: Llama, Qwen3.5, Gemma4 (dense + MoE + resonance variants)."""

from __future__ import annotations

# Delay imports to avoid circular deps
__all__ = [
    "LlamaTransformer",
    "LlamaMoE",
    "ResonantLlama",
    "ResonantLlamaMoE",
    "Qwen3_5Transformer",
    "Qwen3_5MoE",
    "ResonantQwen3_5",
    "ResonantQwen3_5MoE",
    "Gemma4Transformer",
    "Gemma4MoE",
    "ResonantGemma4",
    "ResonantGemma4MoE",
]
