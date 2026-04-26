"""Per-layer configuration for the Resonance Transformer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerConfig:
    """Configuration for a single layer in the Resonance Transformer.

    Overrides the global kernel and/or bias mode for that layer.
    """

    kernel: str
    bias_mode: str
    preset: str | None = None
    n_frequencies: int | None = None


class LayerConfigRegistry:
    """Registry that maps layer-index patterns to :class:`LayerConfig` objects.

    Supported patterns:
        - ``"all"``        → every layer
        - ``"even"``       → layers with even indices (0, 2, 4, …)
        - ``"odd"``        → layers with odd indices (1, 3, 5, …)
        - ``"first:N"``    → first *N* layers (indices ``0 … N-1``)
        - ``"last:N"``     → last *N* layers
        - ``"range:a:b"``  → layers ``a … b-1`` (Python slice semantics)
    """

    def __init__(self, mapping: dict[str, LayerConfig]) -> None:
        self.mapping = mapping

    def get_config(self, layer_index: int, total_layers: int) -> LayerConfig | None:
        """Return the first matching :class:`LayerConfig` for *layer_index*.

        Patterns are evaluated in insertion order.  If no pattern matches,
        ``None`` is returned and the layer falls back to the global config.
        """
        for pattern, config in self.mapping.items():
            if self._matches(pattern, layer_index, total_layers):
                return config
        return None

    @staticmethod
    def _matches(pattern: str, layer_index: int, total_layers: int) -> bool:
        p = pattern.strip().lower()
        if p == "all":
            return True
        if p == "even":
            return layer_index % 2 == 0
        if p == "odd":
            return layer_index % 2 == 1
        if p.startswith("first:"):
            n = int(p.split(":", 1)[1])
            return layer_index < n
        if p.startswith("last:"):
            n = int(p.split(":", 1)[1])
            return layer_index >= total_layers - n
        if p.startswith("range:"):
            parts = p.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid range pattern {pattern!r}; expected range:a:b"
                )
            _, a, b = parts
            return int(a) <= layer_index < int(b)
        raise ValueError(f"Unsupported layer pattern: {pattern!r}")
