"""
Resonance Transformer Architecture
===================================
Two model variants:
  1. StandardTransformer   – vanilla transformer
  2. ResonanceTransformer  – dual embeddings with learnable blend & resonance bias
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standard Transformer
# ---------------------------------------------------------------------------

class StandardTransformer(nn.Module):
    """Vanilla decoder-only transformer for language modeling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, S = tokens.shape
        pos = torch.arange(S, device=tokens.device).unsqueeze(0)

        h = self.embedding(tokens) + self.pos_encoder(pos)
        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=tokens.device), diagonal=1).bool()
        h = self.transformer_decoder(h, h, tgt_mask=mask)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embedding.weight.numel()
            n -= self.pos_encoder.weight.numel()
        return n


# ---------------------------------------------------------------------------
# Resonance Transformer
# ---------------------------------------------------------------------------

class ResonanceTransformer(nn.Module):
    """
    Resonance Transformer with dual semantic/phase embeddings.
    
    attn = softmax( QK^T / sqrt(d) + R * w_r )
    R[i,j] = cos(E_p[i] - E_p[j]).mean()
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        n_frequencies: int = 32,
        use_resonance: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_frequencies = n_frequencies
        self.use_resonance = use_resonance

        # Dual embeddings
        self.semantic_embedding = nn.Embedding(vocab_size, d_model)
        self.phase_embedding = nn.Embedding(vocab_size, n_frequencies)

        # Learnable blend parameter (per dimension)
        self.alpha = nn.Parameter(torch.zeros(d_model))

        # Positional encoding
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        # Resonance attention weight
        self.resonance_weight = nn.Parameter(torch.tensor(1.0))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Projection from phase space to semantic space
        self.phase_proj = nn.Linear(n_frequencies, d_model, bias=False)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie semantic embedding to LM head
        self.lm_head.weight = self.semantic_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize phase embeddings to small random values
        nn.init.normal_(self.phase_embedding.weight, mean=0.0, std=0.01)
        # Initialize alpha near 0.5 (sigmoid(0)=0.5)
        nn.init.zeros_(self.alpha)

    def compute_resonance_matrix(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute resonance matrix R of shape [B, S, S] where
        R[b, i, j] = mean_f cos(phase[b,i,f] - phase[b,j,f])
        """
        phase = self.phase_embedding(tokens)                       # [B, S, F]
        # Expand for pairwise differences
        p_i = phase.unsqueeze(2)                                     # [B, S, 1, F]
        p_j = phase.unsqueeze(1)                                     # [B, 1, S, F]
        R = torch.cos(p_i - p_j).mean(dim=-1)                        # [B, S, S]
        return R

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, S = tokens.shape
        device = tokens.device

        # Semantic & phase streams
        h_sem = self.semantic_embedding(tokens)                      # [B, S, D]
        h_phase_raw = self.phase_embedding(tokens)                   # [B, S, F]
        h_phase = self.phase_proj(h_phase_raw)                       # [B, S, D]

        # Blend with learnable alpha
        alpha = torch.sigmoid(self.alpha)                            # [D]
        h = alpha * h_sem + (1 - alpha) * h_phase

        # Add positional encoding
        pos = torch.arange(S, device=device).unsqueeze(0)
        h = h + self.pos_encoder(pos)

        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()

        if self.use_resonance:
            R = self.compute_resonance_matrix(tokens)                # [B, S, S]
            w_r = torch.sigmoid(self.resonance_weight)
            nhead = self.transformer_decoder.layers[0].self_attn.num_heads
            # Expand R for all heads
            R_expanded = R.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(B * nhead, S, S)
            attn_bias = R_expanded * w_r                              # [B*nhead, S, S]
            # Standard causal mask (True = mask out)
            causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
            additive_mask = attn_bias.clone()
            additive_mask[:, causal_mask] = float('-inf')
            h = self.transformer_decoder(h, h, tgt_mask=additive_mask)
        else:
            h = self.transformer_decoder(h, h, tgt_mask=mask)

        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.semantic_embedding.weight.numel()
            n -= self.phase_embedding.weight.numel()
            n -= self.pos_encoder.weight.numel()
        return n


__all__ = ["StandardTransformer", "ResonanceTransformer"]
