"""Transformer model classes: Standard (vanilla) and Resonance (dual embedding)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ResonanceConfig, StandardConfig
from .kernels import build_kernel
from .phase_embeddings import build_phase_embedding
from .bias_modes import build_bias_mode


# ---------------------------------------------------------------------------
# Initialization presets
# ---------------------------------------------------------------------------

def _preset_values(
    config: ResonanceConfig,
    preset_name: str | None = None,
) -> tuple[float, float, float, bool, bool]:
    """Return (phase_init_std, resonance_attn_weight, resonance_blend,
    normalize_resonance, center_resonance) based on preset."""
    preset = preset_name or config.init_preset
    presets: dict[str, tuple[float, float, float, bool, bool]] = {
        "default":     (0.3, 0.1, 0.3, False, False),
        "wide":        (1.0, 0.3, 0.3, False, False),
        "strong":      (1.2, 1.0, 0.5, False, False),
        "very_strong": (2.0, 2.0, 0.5, False, False),
        "normalized":  (0.3, 0.1, 0.3, True,  False),
    }
    if preset in presets:
        return presets[preset]
    return (
        config.phase_init_std,
        config.resonance_attn_weight,
        config.resonance_blend,
        config.normalize_resonance,
        config.center_resonance,
    )


# ---------------------------------------------------------------------------
# Standard (vanilla) transformer components
# ---------------------------------------------------------------------------

class StandardAttention(nn.Module):
    """Vanilla multi-head causal self-attention.

    Uses a single fused QKV projection followed by split into heads,
    scaled dot-product attention, and an output projection.
    """

    def __init__(self, config: StandardConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.scale = self.head_dim**-0.5
        self.attention_variant = config.attention_variant

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        if self.attention_variant == "alibi":
            slopes = self._build_alibi_slopes(config.n_heads)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        elif self.attention_variant == "standard":
            self.register_buffer("alibi_slopes", torch.empty(0), persistent=False)
        elif self.attention_variant == "deberta_lite":
            raise ValueError("StandardAttention does not implement deberta_lite")
        else:
            raise ValueError(f"unknown attention_variant: {self.attention_variant}")

    @staticmethod
    def _build_alibi_slopes(n_heads: int) -> torch.Tensor:
        """Return ALiBi slopes using the Press et al. construction."""

        def power_of_two_slopes(n: int) -> list[float]:
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = power_of_two_slopes(n_heads)
        else:
            closest = 2 ** math.floor(math.log2(n_heads))
            slopes = power_of_two_slopes(closest)
            extra = power_of_two_slopes(2 * closest)[0::2]
            slopes.extend(extra[: n_heads - closest])
        return torch.tensor(slopes, dtype=torch.float32)

    def _alibi_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        distance = (positions[:, None] - positions[None, :]).clamp(min=0).to(dtype)
        slopes = self.alibi_slopes.to(device=device, dtype=dtype).view(1, self.n_heads, 1, 1)
        return -slopes * distance.view(1, 1, seq_len, seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply standard multi-head self-attention.

        Args:
            x: Input tensor of shape ``(batch, seq_len, embed_dim)``.
            mask: Optional causal mask of shape ``(1, seq_len, seq_len)``
                or broadcastable variant.  ``0`` indicates masked positions.

        Returns:
            Attention output of shape ``(batch, seq_len, embed_dim)``.
        """
        batch_size, seq_len, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.attention_variant == "alibi":
            attn = attn + self._alibi_bias(seq_len, x.device, attn.dtype)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)


class DebertaLiteAttention(nn.Module):
    """Compact DeBERTa-style disentangled relative attention baseline.

    This is not intended as a faithful reproduction of full DeBERTa. It is a
    lightweight comparator for the literature question: does a known
    content/position disentangling trick explain the same gains as the proposed
    structural stream?
    """

    def __init__(self, config: StandardConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.scale = (3 * self.head_dim) ** -0.5
        self.max_seq_len = config.max_seq_len

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.rel_pos = nn.Embedding(2 * config.max_seq_len - 1, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def _relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        rel = positions[:, None] - positions[None, :]
        rel = rel.clamp(min=1 - self.max_seq_len, max=self.max_seq_len - 1)
        return rel + self.max_seq_len - 1

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        rel_ids = self._relative_positions(seq_len, x.device)
        rel = self.rel_pos(rel_ids).reshape(seq_len, seq_len, self.n_heads, self.head_dim)

        content_content = torch.matmul(q, k.transpose(-2, -1))
        content_position = torch.einsum("bhid,ijhd->bhij", q, rel)
        position_content = torch.einsum("ijhd,bhjd->bhij", rel, k)
        attn = (content_content + content_position + position_content) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)


class StandardBlock(nn.Module):
    """A single transformer block for the standard model.

    Pre-norm residual: ``x = x + attn(ln1(x))`` then ``x = x + ff(ln2(x))``.
    """

    def __init__(self, config: StandardConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        if config.attention_variant == "deberta_lite":
            self.attn = DebertaLiteAttention(config)
        else:
            self.attn = StandardAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through one standard transformer block.

        Args:
            x: Input tensor ``(batch, seq_len, embed_dim)``.
            mask: Optional causal attention mask.

        Returns:
            Updated tensor of the same shape.
        """
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class StandardTransformer(nn.Module):
    """Vanilla causal transformer language model (GPT-like).

    Uses token + positional embeddings, a stack of ``StandardBlock`` layers,
    and a final linear language-modelling head tied to the token embedding.
    """

    def __init__(self, config: StandardConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [StandardBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_final = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # Tie weights between input embedding and output projection
        self.lm_head.weight = self.token_embed.weight

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)),
        )

        self.apply(self._init_weights)
        self._init_variant_weights()
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        """Standard GPT-style weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _init_variant_weights(self) -> None:
        """Reset opt-in variant adapters close to identity after global init."""
        for module in self.modules():
            if isinstance(module, ResonanceAttention) and module.phase_condition_qk == "film":
                assert module.q_film is not None and module.k_film is not None
                nn.init.zeros_(module.q_film.weight)
                nn.init.zeros_(module.q_film.bias)
                nn.init.zeros_(module.k_film.weight)
                nn.init.zeros_(module.k_film.bias)
            if isinstance(module, PhaseUpdateBlock):
                last = module.mlp[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Any]:
        """Forward pass and optional next-token loss computation.

        The loss is computed as ``cross_entropy(logits[:, :-1], labels[:, 1:])``
        so that every position predicts the *next* token.

        Args:
            input_ids: Token indices ``(batch, seq_len)``.
            labels: Optional target token indices of the same shape.  If
                provided, the ``"loss"`` key is included in the returned dict.

        Returns:
            Dictionary with at least ``"logits"`` (shape
            ``(batch, seq_len, vocab_size)``).  Optionally ``"loss"``.
        """
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        hidden = self.ln_final(x)
        logits = self.lm_head(hidden)

        result: dict[str, Any] = {"logits": logits}
        if return_hidden:
            result["hidden_states"] = hidden
        if labels is not None:
            # Shift so that each position predicts the next token
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result


# ---------------------------------------------------------------------------
# Resonance transformer components
# ---------------------------------------------------------------------------

class ResonanceEmbedding(nn.Module):
    """Dual embedding combining semantic and phase representations.

    Each token receives:
    1. A *semantic* embedding (standard, learned).
    2. A *phase* embedding (structural, learned, lower-dimensional).
    3. A *learnable blend* parameter (sigmoid-gated) controlling the
       interpolation between semantic and projected phase embeddings.

    Phase embeddings may be initialised randomly or from phonetic (rhyme)
    structure via :func:`_init_phonetic_phases`.
    """

    def __init__(
        self,
        config: ResonanceConfig,
        rhyme_index: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        # Apply initialization preset
        phase_init_std, attn_weight, blend, norm_res, center_res = _preset_values(config)

        # Semantic embedding (standard)
        self.semantic = nn.Embedding(config.vocab_size, config.embed_dim)

        # Phase embedding (structural / phonetic) — modular, swappable
        self.phase = build_phase_embedding(
            config.phase_embedding,
            config.vocab_size,
            config.n_frequencies,
            init_std=phase_init_std,
            rank=config.phase_embedding_rank,
            n_scales=config.phase_embedding_scales,
        )
        self.phase_proj = nn.Linear(config.n_frequencies, config.embed_dim, bias=False)

        # Learnable blend parameter (per-dimension interpolation)
        self.blend = nn.Parameter(
            torch.full((config.embed_dim,), blend)
        )

        # Positional encoding
        self.position = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Resonance kernel (modular, swappable)
        self.kernel = build_kernel(
            config.resonance_kernel,
            config.n_frequencies,
            gamma=config.kernel_gamma,
            learnable_gamma=config.kernel_learnable_gamma,
            rank=config.kernel_rank,
            temperature=config.kernel_temperature,
            walk_atoms=getattr(config, "walk_atoms", 5),
            use_chiral=getattr(config, "walk_use_chiral", True),
            walk_band=getattr(config, "walk_band", 8),
            walk_use_phase_drive=getattr(config, "walk_use_phase_drive", True),
            walk_atom_set=getattr(config, "walk_atom_set", "v1"),
        )

        # Store effective init values for attention module
        self._effective_phase_init_std = phase_init_std
        self._effective_attn_weight = attn_weight
        self._effective_normalize = norm_res
        self._effective_center = center_res

        # Initialise
        self._init_weights()
        if config.phonetic_init and rhyme_index is not None:
            self._init_phonetic_phases(rhyme_index)
        else:
            self._init_random_phases()

    def _init_weights(self) -> None:
        """Initialise semantic, projection, and positional embeddings."""
        nn.init.normal_(self.semantic.weight, std=0.02)
        nn.init.normal_(self.phase_proj.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)

    def _init_random_phases(self) -> None:
        """Initialise phase embeddings from a Gaussian distribution."""
        # Only applies to learned phase embeddings with explicit weight/angle params
        from .phase_embeddings import RealPhaseEmbedding, ComplexAnglePhaseEmbedding
        if isinstance(self.phase, RealPhaseEmbedding):
            nn.init.normal_(self.phase.weight, std=self._effective_phase_init_std)
        elif isinstance(self.phase, ComplexAnglePhaseEmbedding):
            nn.init.normal_(self.phase.angle, std=self._effective_phase_init_std)
        # FourierFixed and Hierarchical handle their own init

    def _init_phonetic_phases(self, rhyme_index: dict[str, Any]) -> None:
        """Initialise phase embeddings so rhyming words are close in phase space.

        Only supported for ``real`` phase embeddings currently.
        """
        from .phase_embeddings import RealPhaseEmbedding
        if not isinstance(self.phase, RealPhaseEmbedding):
            return  # Skip for non-real phase embeddings

        rhyme_to_tokens: dict[str, list[int]] = rhyme_index.get("rhyme_to_tokens", {})

        # Assign a base phase vector to each rhyme group
        rhyme_to_phase = {
            rhyme: torch.randn(self.config.n_frequencies) * 0.3
            for rhyme in rhyme_to_tokens.keys()
        }

        with torch.no_grad():
            self.phase.weight.data = torch.randn_like(self.phase.weight) * 0.3

            phonetic_count = 0
            for rhyme, tokens in rhyme_to_tokens.items():
                base_phase = rhyme_to_phase[rhyme]
                for token_id in tokens:
                    self.phase.weight.data[token_id] = (
                        base_phase + torch.randn(self.config.n_frequencies) * 0.03
                    )
                    phonetic_count += 1

        self.phonetic_count = phonetic_count

    def get_phases(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return raw phase embeddings for per-layer kernel overrides.

        Args:
            token_ids: Integer token ids ``(batch, seq_len)``.

        Returns:
            Phase tensor ``(batch, seq_len, n_frequencies)``.
        """
        return self.phase(token_ids)

    def get_resonance_matrix(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute pairwise resonance scores via the configured kernel.

        Args:
            token_ids: Integer token ids ``(batch, seq_len)``.

        Returns:
            Resonance matrix ``(batch, seq_len, seq_len)``.
        """
        if not (self.config.use_phase_stream or self.config.use_resonance_bias):
            batch_size, seq_len = token_ids.shape
            return torch.zeros(batch_size, seq_len, seq_len, device=token_ids.device)
        phases = self.phase(token_ids)
        return self.kernel(phases)

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce blended embeddings and the resonance matrix.

        Args:
            token_ids: Token indices ``(batch, seq_len)``.

        Returns:
            A tuple of ``(embeddings, resonance)`` where *embeddings* has
            shape ``(batch, seq_len, embed_dim)`` and *resonance* has shape
            ``(batch, seq_len, seq_len)``.
        """
        batch_size, seq_len = token_ids.shape

        sem = self.semantic(token_ids)

        if self.config.use_phase_stream:
            ph = self.phase(token_ids)
            ph_proj = self.phase_proj(ph)
            blend = torch.sigmoid(self.blend)
            embeddings = (1 - blend) * sem + blend * ph_proj
        else:
            embeddings = sem

        positions = torch.arange(seq_len, device=token_ids.device)
        embeddings = embeddings + self.position(positions)

        resonance = self.get_resonance_matrix(token_ids)
        embeddings = self.dropout(embeddings)

        return embeddings, resonance


class ResonanceAttention(nn.Module):
    """Multi-head causal self-attention with a resonance bias.

    In addition to the standard scaled dot-product term, the attention
    logits receive an additive bias proportional to the pairwise resonance
    scores.  The per-head weighting of this bias is learnable.
    """

    def __init__(
        self,
        config: ResonanceConfig,
        bias_mode: Any | None = None,
        preset: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.scale = self.head_dim**-0.5
        self.attention_variant = getattr(config, "attention_variant", "standard")

        # Apply preset for effective attention weight
        _, attn_weight, _, norm_res, center_res = _preset_values(config, preset)
        self._effective_normalize = norm_res
        self._effective_center = center_res

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.resonance_weight = nn.Parameter(
            torch.full((config.n_heads,), attn_weight)
        )
        n_structural = max(0, min(config.n_structural_heads, config.n_heads))
        self.n_structural_heads = n_structural
        if n_structural > 0:
            self.structural_head_weight = nn.Parameter(
                torch.full((n_structural,), config.structural_head_scale)
            )
        else:
            self.register_parameter("structural_head_weight", None)
        self.dropout = nn.Dropout(config.dropout)
        if self.attention_variant == "alibi":
            slopes = StandardAttention._build_alibi_slopes(config.n_heads)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        elif self.attention_variant == "standard":
            self.register_buffer("alibi_slopes", torch.empty(0), persistent=False)
        else:
            raise ValueError(f"unknown resonance attention_variant: {self.attention_variant}")

        self.relation_value_mode = config.relation_value_mode
        if self.relation_value_mode == "additive":
            self.relation_value_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
            self.relation_value_gate = nn.Parameter(torch.tensor(0.0))
        elif self.relation_value_mode == "none":
            self.relation_value_proj = None
            self.register_parameter("relation_value_gate", None)
        else:
            raise ValueError(f"unknown relation_value_mode: {self.relation_value_mode}")

        self.phase_condition_qk = config.phase_condition_qk
        if self.phase_condition_qk == "film":
            self.q_film = nn.Linear(config.n_frequencies, 2 * config.embed_dim)
            self.k_film = nn.Linear(config.n_frequencies, 2 * config.embed_dim)
            nn.init.zeros_(self.q_film.weight)
            nn.init.zeros_(self.q_film.bias)
            nn.init.zeros_(self.k_film.weight)
            nn.init.zeros_(self.k_film.bias)
        elif self.phase_condition_qk == "none":
            self.q_film = None
            self.k_film = None
        else:
            raise ValueError(f"unknown phase_condition_qk: {self.phase_condition_qk}")

        # Bias application mode (modular, swappable)
        if bias_mode is not None:
            self.bias_mode = bias_mode
        else:
            self.bias_mode = build_bias_mode(
                config.bias_mode,
                n_heads=config.n_heads,
                gate_init=config.bias_gate_init,
            )

    def _prepare_resonance_bias(
        self,
        resonance: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not (self._effective_center or self._effective_normalize):
            return resonance
        if mask is None:
            centered = resonance - resonance.mean(dim=-1, keepdim=True)
            if not self._effective_normalize:
                return centered
            std = (centered**2).mean(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
            return centered / std
        allowed = mask.to(device=resonance.device, dtype=resonance.dtype)
        while allowed.dim() > resonance.dim():
            allowed = allowed.squeeze(0)
        denom = allowed.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (resonance * allowed).sum(dim=-1, keepdim=True) / denom
        centered = (resonance - mean) * allowed
        if not self._effective_normalize:
            return centered
        variance = ((centered**2) * allowed).sum(dim=-1, keepdim=True) / denom
        std = variance.clamp(min=1e-8).sqrt()
        return (centered / std) * allowed

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor,
        mask: torch.Tensor | None = None,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply resonance-biased multi-head attention.

        Args:
            x: Input tensor ``(batch, seq_len, embed_dim)``.
            resonance: Pairwise resonance matrix ``(batch, seq_len, seq_len)``.
            mask: Optional causal mask broadcastable to
                ``(batch, n_heads, seq_len, seq_len)``.

        Returns:
            Attention output ``(batch, seq_len, embed_dim)``.
        """
        batch_size, seq_len, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.phase_condition_qk == "film":
            if phases is None:
                raise ValueError("phases are required for phase_condition_qk='film'")
            q_mod = self.q_film(phases).reshape(
                batch_size, seq_len, 2, self.n_heads, self.head_dim
            )
            k_mod = self.k_film(phases).reshape(
                batch_size, seq_len, 2, self.n_heads, self.head_dim
            )
            q_gamma, q_beta = q_mod[:, :, 0], q_mod[:, :, 1]
            k_gamma, k_beta = k_mod[:, :, 0], k_mod[:, :, 1]
            q_gamma = q_gamma.permute(0, 2, 1, 3)
            q_beta = q_beta.permute(0, 2, 1, 3)
            k_gamma = k_gamma.permute(0, 2, 1, 3)
            k_beta = k_beta.permute(0, 2, 1, 3)
            q = q * (1.0 + q_gamma) + q_beta
            k = k * (1.0 + k_gamma) + k_beta

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.attention_variant == "alibi":
            positions = torch.arange(seq_len, device=x.device)
            distance = (positions[:, None] - positions[None, :]).clamp(min=0).to(attn.dtype)
            slopes = self.alibi_slopes.to(device=x.device, dtype=attn.dtype).view(
                1, self.n_heads, 1, 1
            )
            attn = attn - slopes * distance.view(1, 1, seq_len, seq_len)
        prepared_resonance: torch.Tensor | None = None

        if self.n_structural_heads > 0:
            prepared_resonance = self._prepare_resonance_bias(resonance, mask)
            structural_logits = prepared_resonance.unsqueeze(1) * self.structural_head_weight.view(
                1, self.n_structural_heads, 1, 1
            )
            attn = attn.clone()
            attn[:, : self.n_structural_heads] = structural_logits

        # Apply resonance bias via modular bias mode
        if self.config.use_resonance_bias:
            resonance_bias = prepared_resonance
            if resonance_bias is None:
                resonance_bias = self._prepare_resonance_bias(resonance, mask)
            resonance_bias = resonance_bias.unsqueeze(1) * self.resonance_weight.view(
                1, self.n_heads, 1, 1
            )
            if self.n_structural_heads > 0:
                if self.n_structural_heads < self.n_heads:
                    attn_tail = self.bias_mode(
                        attn[:, self.n_structural_heads :],
                        resonance_bias[:, self.n_structural_heads :],
                        mask,
                    )
                    attn = torch.cat([attn[:, : self.n_structural_heads], attn_tail], dim=1)
            else:
                attn = self.bias_mode(attn, resonance_bias, mask)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        if self.relation_value_mode == "additive":
            relation_logits = self._prepare_resonance_bias(resonance, mask)
            if mask is not None:
                relation_logits = relation_logits.masked_fill(mask.squeeze(0) == 0, float("-inf"))
            relation_weights = F.softmax(relation_logits, dim=-1)
            relation_values = self.relation_value_proj(x)
            relation_context = torch.matmul(relation_weights, relation_values)
            out = out + torch.tanh(self.relation_value_gate) * relation_context
        return self.out_proj(out)


class PhaseUpdateBlock(nn.Module):
    """Layerwise update for the compact phase/structural state."""

    def __init__(self, config: ResonanceConfig) -> None:
        super().__init__()
        self.config = config
        self.mode = config.phase_update_mode
        self.scale = config.phase_update_scale
        hidden = max(config.n_frequencies, config.n_frequencies * config.phase_update_hidden_mult)
        self.ln = nn.LayerNorm(config.n_frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_frequencies, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_frequencies),
        )
        if self.mode == "self_attn":
            n_heads = max(1, min(config.phase_update_heads, config.n_frequencies))
            while config.n_frequencies % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            self.attn_ln = nn.LayerNorm(config.n_frequencies)
            self.attn = nn.MultiheadAttention(
                embed_dim=config.n_frequencies,
                num_heads=n_heads,
                dropout=config.dropout,
                batch_first=True,
            )
        elif self.mode in {"none", "mlp"}:
            self.attn_ln = None
            self.attn = None
        else:
            raise ValueError(f"unknown phase_update_mode: {self.mode}")

    def forward(
        self,
        phases: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "none":
            return phases
        updated = phases
        if self.mode == "self_attn":
            attn_mask = None
            if mask is not None:
                # nn.MultiheadAttention uses True for disallowed positions.
                attn_mask = (mask.squeeze(0) == 0).to(device=phases.device)
            attn_input = self.attn_ln(updated)
            attn_out, _ = self.attn(
                attn_input,
                attn_input,
                attn_input,
                attn_mask=attn_mask,
                need_weights=False,
            )
            updated = updated + self.scale * attn_out
        updated = updated + self.scale * self.mlp(self.ln(updated))
        return updated


class ResonanceBlock(nn.Module):
    """A single transformer block for the Resonance model.

    Identical structure to ``StandardBlock`` except that the attention
    sub-layer uses ``ResonanceAttention`` and therefore requires the
    *resonance* matrix as an additional input.
    """

    def __init__(
        self,
        config: ResonanceConfig,
        kernel: Any | None = None,
        bias_mode: Any | None = None,
        preset: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = ResonanceAttention(config, bias_mode=bias_mode, preset=preset)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )
        self.kernel = kernel  # Optional per-layer kernel override
        self.dynamic_kernel = (
            build_kernel(
                config.resonance_kernel,
                config.n_frequencies,
                gamma=config.kernel_gamma,
                learnable_gamma=config.kernel_learnable_gamma,
                rank=config.kernel_rank,
                temperature=config.kernel_temperature,
            )
            if config.phase_update_mode != "none"
            else None
        )
        self.phase_update = (
            PhaseUpdateBlock(config) if config.phase_update_mode != "none" else None
        )

    def forward(
        self,
        x: torch.Tensor,
        resonance: torch.Tensor,
        mask: torch.Tensor | None = None,
        phases: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through one resonance transformer block.

        Args:
            x: Input tensor ``(batch, seq_len, embed_dim)``.
            resonance: Pairwise resonance matrix ``(batch, seq_len, seq_len)``.
            mask: Optional causal attention mask.
            phases: Optional phase embeddings ``(batch, seq_len, n_frequencies)``.
                Required when this block has a per-layer kernel override.

        Returns:
            Updated tensor of the same shape.
        """
        if phases is not None and self.phase_update is not None:
            phases = self.phase_update(phases, mask)
        if self.kernel is not None and phases is not None:
            resonance = self.kernel(phases)
        elif self.dynamic_kernel is not None and phases is not None:
            resonance = self.dynamic_kernel(phases)
        x = x + self.attn(self.ln1(x), resonance, mask, phases)
        x = x + self.ff(self.ln2(x))
        return x, phases


class ResonanceTransformer(nn.Module):
    """Full Resonance transformer language model.

    Uses :class:`ResonanceEmbedding` to produce dual semantic/phase
    representations, applies a stack of :class:`ResonanceBlock` layers,
    and projects back to vocabulary logits.  The output projection is
    tied to the *semantic* embedding weights.
    """

    def __init__(
        self,
        config: ResonanceConfig,
        rhyme_index: dict[str, Any] | None = None,
        layer_configs: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.embedding = ResonanceEmbedding(config, rhyme_index)

        blocks: list[ResonanceBlock] = []
        for i in range(config.n_layers):
            kwargs: dict[str, Any] = {}
            if layer_configs is not None:
                from .layer_config import LayerConfigRegistry

                lc = layer_configs.get_config(i, config.n_layers)
                if lc is not None:
                    n_freq = (
                        lc.n_frequencies
                        if lc.n_frequencies is not None
                        else config.n_frequencies
                    )
                    kwargs["kernel"] = build_kernel(
                        lc.kernel,
                        n_freq,
                        gamma=config.kernel_gamma,
                        learnable_gamma=config.kernel_learnable_gamma,
                        rank=config.kernel_rank,
                        temperature=config.kernel_temperature,
                    )
                    kwargs["bias_mode"] = build_bias_mode(
                        lc.bias_mode,
                        n_heads=config.n_heads,
                        gate_init=config.bias_gate_init,
                    )
                    if lc.preset is not None:
                        kwargs["preset"] = lc.preset
            blocks.append(ResonanceBlock(config, **kwargs))

        self.blocks = nn.ModuleList(blocks)

        self.ln_final = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.semantic.weight

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)),
        )

        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        """Standard GPT-style weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, Any]:
        """Forward pass and optional next-token loss computation.

        Args:
            input_ids: Token indices ``(batch, seq_len)``.
            labels: Optional target token indices of the same shape.

        Returns:
            Dictionary with at least ``"logits"``.  Optionally ``"loss"``.
        """
        batch_size, seq_len = input_ids.shape

        x, resonance = self.embedding(input_ids)
        phases = self.embedding.get_phases(input_ids)
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0)

        for block in self.blocks:
            x, phases = block(x, resonance, mask, phases)

        hidden = self.ln_final(x)
        logits = self.lm_head(hidden)

        result: dict[str, Any] = {"logits": logits}
        if return_hidden:
            result["hidden_states"] = hidden
            result["phase_states"] = phases
            result["resonance_matrix"] = resonance
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return result
