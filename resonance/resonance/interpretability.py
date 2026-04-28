"""Resonance-native tracing and intervention helpers.

The normal model ``forward`` path keeps the implementation compact, but
interpretability work needs to see the separate streams and the attention-logit
decomposition.  This module replays the forward pass with explicit captures:

    QK logits
    resonance logits
    combined logits
    attention probabilities
    attention delta caused by resonance
    semantic / phase / projected phase streams

The functions are deliberately small and framework-light so they can be used in
experiments before integrating with TransformerLens/SAELens-style tooling.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from .models import ResonanceTransformer, StandardTransformer


@dataclass
class AttentionTrace:
    """Per-layer attention decomposition."""

    qk_logits: torch.Tensor
    resonance_logits: torch.Tensor
    combined_logits: torch.Tensor
    qk_attention: torch.Tensor
    combined_attention: torch.Tensor
    attention_delta: torch.Tensor
    hidden_pre: torch.Tensor
    hidden_post: torch.Tensor


@dataclass
class ResonanceTrace:
    """Full trace for one resonance-model forward pass."""

    input_ids: torch.Tensor
    semantic: torch.Tensor
    phase: torch.Tensor
    phase_projected: torch.Tensor | None
    blend: torch.Tensor | None
    token_embeddings: torch.Tensor
    resonance_matrix: torch.Tensor
    layers: list[AttentionTrace]
    hidden_states: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor | None


@dataclass
class StandardTrace:
    """Minimal trace for a standard transformer forward pass."""

    input_ids: torch.Tensor
    token_embeddings: torch.Tensor
    layers: list[AttentionTrace]
    hidden_states: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor | None


def _causal_mask(model: ResonanceTransformer | StandardTransformer, seq_len: int) -> torch.Tensor:
    return model.causal_mask[:seq_len, :seq_len].unsqueeze(0)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        logits = logits.masked_fill(mask == 0, float("-inf"))
    return F.softmax(logits, dim=-1)


def _lm_loss(logits: torch.Tensor, labels: torch.Tensor | None, vocab_size: int) -> torch.Tensor | None:
    if labels is None:
        return None
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, vocab_size),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def trace_resonance_forward(
    model: ResonanceTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> ResonanceTrace:
    """Replay a resonance forward pass with stream and attention captures."""
    model.eval()
    batch_size, seq_len = input_ids.shape
    mask = _causal_mask(model, seq_len)
    positions = torch.arange(seq_len, device=input_ids.device)

    embedding = model.embedding
    semantic = embedding.semantic(input_ids)
    phase = embedding.phase(input_ids)
    phase_projected = None
    blend = None
    if model.config.use_phase_stream:
        phase_projected = embedding.phase_proj(phase)
        blend = torch.sigmoid(embedding.blend)
        x = (1 - blend) * semantic + blend * phase_projected
    else:
        x = semantic
    x = embedding.dropout(x + embedding.position(positions))
    token_embeddings = x.detach().cpu()
    resonance = embedding.get_resonance_matrix(input_ids)

    layer_traces: list[AttentionTrace] = []
    for block in model.blocks:
        hidden_pre = x
        x_norm = block.ln1(x)
        attn_mod = block.attn
        qkv = (
            attn_mod.qkv(x_norm)
            .reshape(batch_size, seq_len, 3, attn_mod.n_heads, attn_mod.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        qk_logits = torch.matmul(q, k.transpose(-2, -1)) * attn_mod.scale
        if model.config.use_resonance_bias:
            resonance_bias = attn_mod._prepare_resonance_bias(resonance, mask)
            resonance_logits = resonance_bias.unsqueeze(1) * attn_mod.resonance_weight.view(
                1, attn_mod.n_heads, 1, 1
            )
        else:
            resonance_logits = torch.zeros_like(qk_logits)
        combined_logits = qk_logits + resonance_logits
        qk_attention = _masked_softmax(qk_logits, mask)
        combined_attention = _masked_softmax(combined_logits, mask)
        attn = attn_mod.dropout(combined_attention)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        x = x + attn_mod.out_proj(attn_out)
        x = x + block.ff(block.ln2(x))
        layer_traces.append(
            AttentionTrace(
                qk_logits=qk_logits.detach().cpu(),
                resonance_logits=resonance_logits.detach().cpu(),
                combined_logits=combined_logits.detach().cpu(),
                qk_attention=qk_attention.detach().cpu(),
                combined_attention=combined_attention.detach().cpu(),
                attention_delta=(combined_attention - qk_attention).detach().cpu(),
                hidden_pre=hidden_pre.detach().cpu(),
                hidden_post=x.detach().cpu(),
            )
        )

    hidden = model.ln_final(x)
    logits = model.lm_head(hidden)
    loss = _lm_loss(logits, labels, model.config.vocab_size)
    return ResonanceTrace(
        input_ids=input_ids.detach().cpu(),
        semantic=semantic.detach().cpu(),
        phase=phase.detach().cpu(),
        phase_projected=None if phase_projected is None else phase_projected.detach().cpu(),
        blend=None if blend is None else blend.detach().cpu(),
        token_embeddings=token_embeddings,
        resonance_matrix=resonance.detach().cpu(),
        layers=layer_traces,
        hidden_states=hidden.detach().cpu(),
        logits=logits.detach().cpu(),
        loss=None if loss is None else loss.detach().cpu(),
    )


@torch.no_grad()
def trace_standard_forward(
    model: StandardTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> StandardTrace:
    """Replay a standard forward pass with comparable attention captures."""
    model.eval()
    batch_size, seq_len = input_ids.shape
    mask = _causal_mask(model, seq_len)
    positions = torch.arange(seq_len, device=input_ids.device)
    x = model.dropout(model.token_embed(input_ids) + model.pos_embed(positions))
    token_embeddings = x

    layer_traces: list[AttentionTrace] = []
    for block in model.blocks:
        hidden_pre = x
        x_norm = block.ln1(x)
        attn_mod = block.attn
        qkv = (
            attn_mod.qkv(x_norm)
            .reshape(batch_size, seq_len, 3, attn_mod.n_heads, attn_mod.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        qk_logits = torch.matmul(q, k.transpose(-2, -1)) * attn_mod.scale
        combined_logits = qk_logits
        if getattr(attn_mod, "attention_variant", None) == "alibi":
            combined_logits = combined_logits + attn_mod._alibi_bias(
                seq_len,
                input_ids.device,
                combined_logits.dtype,
            )
        elif hasattr(attn_mod, "rel_pos"):
            rel_ids = attn_mod._relative_positions(seq_len, input_ids.device)
            rel = attn_mod.rel_pos(rel_ids).reshape(
                seq_len,
                seq_len,
                attn_mod.n_heads,
                attn_mod.head_dim,
            )
            content_position = torch.einsum("bhid,ijhd->bhij", q, rel)
            position_content = torch.einsum("ijhd,bhjd->bhij", rel, k)
            combined_logits = qk_logits + (content_position + position_content) * attn_mod.scale
        combined_attention = _masked_softmax(combined_logits, mask)
        attn = attn_mod.dropout(combined_attention)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        x = x + attn_mod.out_proj(attn_out)
        x = x + block.ff(block.ln2(x))
        zeros = torch.zeros_like(qk_logits)
        layer_traces.append(
            AttentionTrace(
                qk_logits=qk_logits.detach().cpu(),
                resonance_logits=zeros.detach().cpu(),
                combined_logits=combined_logits.detach().cpu(),
                qk_attention=_masked_softmax(qk_logits, mask).detach().cpu(),
                combined_attention=combined_attention.detach().cpu(),
                attention_delta=(combined_attention - _masked_softmax(qk_logits, mask)).detach().cpu(),
                hidden_pre=hidden_pre.detach().cpu(),
                hidden_post=x.detach().cpu(),
            )
        )

    hidden = model.ln_final(x)
    logits = model.lm_head(hidden)
    loss = _lm_loss(logits, labels, model.config.vocab_size)
    return StandardTrace(
        input_ids=input_ids.detach().cpu(),
        token_embeddings=token_embeddings.detach().cpu(),
        layers=layer_traces,
        hidden_states=hidden.detach().cpu(),
        logits=logits.detach().cpu(),
        loss=None if loss is None else loss.detach().cpu(),
    )


def effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy effective rank for a 2D matrix."""
    singular_values = torch.linalg.svdvals(matrix.float())
    singular_values = singular_values[singular_values > eps]
    if singular_values.numel() == 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    entropy = -(probs * torch.log(probs + eps)).sum()
    return float(torch.exp(entropy).item())


def resonance_diagnostics(trace: ResonanceTrace) -> dict[str, float]:
    """Summarise resonance and attention-delta behavior for a trace."""
    r = trace.resonance_matrix.float()
    first = r[0]
    off_diag = first.clone()
    off_diag.fill_diagonal_(0)
    delta_abs = [
        float(layer.attention_delta.abs().mean().item()) for layer in trace.layers
    ]
    resonance_logit_abs = [
        float(layer.resonance_logits.abs().mean().item()) for layer in trace.layers
    ]
    return {
        "resonance_mean": float(r.mean().item()),
        "resonance_std": float(r.std(unbiased=False).item()),
        "resonance_min": float(r.min().item()),
        "resonance_max": float(r.max().item()),
        "resonance_effective_rank_first": effective_rank(first),
        "offdiag_abs_mean_first": float(off_diag.abs().mean().item()),
        "attention_delta_abs_mean": float(sum(delta_abs) / max(len(delta_abs), 1)),
        "resonance_logit_abs_mean": float(
            sum(resonance_logit_abs) / max(len(resonance_logit_abs), 1)
        ),
    }


@contextmanager
def phase_intervention(
    model: ResonanceTransformer,
    *,
    mode: str,
    sigma: float = 0.1,
    seed: int = 0,
) -> Iterator[None]:
    """Temporarily perturb the phase embedding table.

    Modes:
    - ``zero``: set phase embeddings to zero.
    - ``permute``: permute token-to-phase assignment.
    - ``noise``: add Gaussian noise scaled by embedding std.
    """
    original = model.embedding.phase.weight.detach().clone()
    generator = torch.Generator(device=original.device)
    generator.manual_seed(seed)
    with torch.no_grad():
        if mode == "zero":
            model.embedding.phase.weight.zero_()
        elif mode == "permute":
            perm = torch.randperm(original.size(0), generator=generator, device=original.device)
            model.embedding.phase.weight.copy_(original[perm])
        elif mode == "noise":
            std = original.std(unbiased=False).clamp(min=1e-6)
            noise = torch.randn(
                original.shape,
                generator=generator,
                device=original.device,
                dtype=original.dtype,
            )
            model.embedding.phase.weight.copy_(original + sigma * std * noise)
        else:
            raise ValueError(f"unknown phase intervention mode: {mode}")
    try:
        yield
    finally:
        with torch.no_grad():
            model.embedding.phase.weight.copy_(original)


@contextmanager
def resonance_weight_scale(
    model: ResonanceTransformer,
    scale: float,
) -> Iterator[None]:
    """Temporarily scale all per-head resonance attention weights."""
    originals = [block.attn.resonance_weight.detach().clone() for block in model.blocks]
    with torch.no_grad():
        for block in model.blocks:
            block.attn.resonance_weight.mul_(scale)
    try:
        yield
    finally:
        with torch.no_grad():
            for block, original in zip(model.blocks, originals, strict=True):
                block.attn.resonance_weight.copy_(original)


@contextmanager
def phase_rows_patch(
    model: ResonanceTransformer,
    token_ids: torch.Tensor | list[int],
    replacement_rows: torch.Tensor,
) -> Iterator[None]:
    """Temporarily replace selected rows of the phase embedding table.

    This is the simplest causal-patching primitive for token-level phase
    experiments.  ``replacement_rows`` must have shape
    ``(len(token_ids), n_frequencies)`` and is copied into
    ``model.embedding.phase.weight[token_ids]`` for the duration of the context.
    """
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=model.embedding.phase.weight.device)
    replacement = replacement_rows.to(
        device=model.embedding.phase.weight.device,
        dtype=model.embedding.phase.weight.dtype,
    )
    selected = model.embedding.phase.weight.index_select(0, ids)
    if replacement.shape != selected.shape:
        raise ValueError(
            "replacement_rows must match selected phase rows: "
            f"got {tuple(replacement.shape)}, expected {tuple(selected.shape)}"
        )
    original = selected.detach().clone()
    with torch.no_grad():
        model.embedding.phase.weight.index_copy_(0, ids, replacement)
    try:
        yield
    finally:
        with torch.no_grad():
            model.embedding.phase.weight.index_copy_(0, ids, original)


@contextmanager
def phase_rows_copy(
    model: ResonanceTransformer,
    *,
    source_token_ids: torch.Tensor | list[int],
    target_token_ids: torch.Tensor | list[int],
) -> Iterator[None]:
    """Temporarily copy phase rows from source tokens onto target tokens."""
    source_ids = torch.as_tensor(
        source_token_ids,
        dtype=torch.long,
        device=model.embedding.phase.weight.device,
    )
    target_ids = torch.as_tensor(
        target_token_ids,
        dtype=torch.long,
        device=model.embedding.phase.weight.device,
    )
    if source_ids.numel() != target_ids.numel():
        raise ValueError("source_token_ids and target_token_ids must have the same length")
    replacement = model.embedding.phase.weight[source_ids].detach().clone()
    with phase_rows_patch(model, target_ids, replacement):
        yield
