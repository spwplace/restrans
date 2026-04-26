"""Sparse Autoencoder (SAE) infrastructure for Resonance Transformer interpretability."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ResonanceTransformer


# ---------------------------------------------------------------------------
# Sparse Autoencoder
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """A simple sparse autoencoder with L1 sparsity penalty.

    Architecture:
        encoder: input_dim -> hidden_dim (ReLU)
        decoder: hidden_dim -> input_dim

    The hidden activations are encouraged to be sparse via an L1 penalty
    applied during training.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialise weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode a batch of activation vectors.

        Args:
            x: Input activations of shape ``(..., input_dim)``.

        Returns:
            A tuple of ``(reconstruction, hidden_activations)`` where
            *hidden_activations* has shape ``(..., hidden_dim)``.
        """
        h = F.relu(self.encoder(x))
        recon = self.decoder(h)
        return recon, h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse hidden activations for *x*."""
        with torch.no_grad():
            return F.relu(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from hidden activations."""
        with torch.no_grad():
            return self.decoder(h)


# ---------------------------------------------------------------------------
# SAE Trainer
# ---------------------------------------------------------------------------

class SAETrainer:
    """Trainer for a :class:`SparseAutoencoder`.

    Minimises ``MSE(reconstruction, target) + λ * mean(|hidden|)``.
    Tracks loss components, mean activation, and dead-feature ratio.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        lr: float = 1e-3,
        l1_lambda: float = 1e-3,
        device: torch.device | str | None = None,
    ) -> None:
        from .device import get_device

        self.sae = sae
        self.lr = lr
        self.l1_lambda = l1_lambda
        self.device = get_device(device)
        self.sae.to(self.device)

        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

        self.history: dict[str, list[float]] = {
            "loss": [],
            "recon_mse": [],
            "sparsity_penalty": [],
            "mean_activation": [],
            "dead_features": [],
        }

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """Perform one gradient step on a batch of activations.

        Args:
            batch: Activations of shape ``(batch_size, input_dim)`` or
                ``(batch_size, seq_len, input_dim)``.

        Returns:
            Dictionary of scalar metrics for this step.
        """
        batch = batch.to(self.device)
        self.sae.train()
        self.optimizer.zero_grad()

        recon, h = self.sae(batch)
        recon_mse = F.mse_loss(recon, batch)
        sparsity = h.abs().mean()
        loss = recon_mse + self.l1_lambda * sparsity

        loss.backward()
        self.optimizer.step()

        # Dead features: never activated across this batch
        dead = (h.sum(dim=0) == 0).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "recon_mse": recon_mse.item(),
            "sparsity_penalty": sparsity.item(),
            "mean_activation": h.mean().item(),
            "dead_features": dead,
        }
        return metrics

    def train_epoch(
        self,
        activations: torch.Tensor,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> dict[str, float]:
        """Train for one epoch on a tensor of activations.

        Args:
            activations: Tensor of shape ``(N, input_dim)`` or
                ``(N, seq_len, input_dim)``.  If 3-D it is flattened to
                ``(N * seq_len, input_dim)``.
            batch_size: Number of samples per gradient step.
            shuffle: Whether to shuffle the data.

        Returns:
            Mean metrics across the epoch.
        """
        if activations.dim() == 3:
            activations = activations.view(-1, activations.shape[-1])

        n_samples = activations.shape[0]
        indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)

        epoch_metrics: dict[str, list[float]] = {
            "loss": [],
            "recon_mse": [],
            "sparsity_penalty": [],
            "mean_activation": [],
            "dead_features": [],
        }

        for i in range(0, n_samples, batch_size):
            idx = indices[i : i + batch_size]
            batch = activations[idx]
            step_metrics = self.train_step(batch)
            for k, v in step_metrics.items():
                epoch_metrics[k].append(v)

        mean_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        for k, v in mean_metrics.items():
            self.history[k].append(v)
        return mean_metrics

    def save_checkpoint(self, path: str) -> None:
        """Save SAE state and optimizer state to *path*."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "sae_state_dict": self.sae.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "lr": self.lr,
                "l1_lambda": self.l1_lambda,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load SAE state and optimizer state from *path*."""
        ckpt = torch.load(path, map_location=self.device)
        self.sae.load_state_dict(ckpt["sae_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.lr = ckpt.get("lr", self.lr)
        self.l1_lambda = ckpt.get("l1_lambda", self.l1_lambda)


# ---------------------------------------------------------------------------
# Stream Extractor
# ---------------------------------------------------------------------------

class StreamExtractor:
    """Extract named activation streams from a ``ResonanceTransformer``.

    Uses forward hooks to capture intermediate activations without modifying
    the model's forward signature.
    """

    def __init__(self, model: ResonanceTransformer) -> None:
        self.model = model
        self._handles: list = []
        self._cache: dict[str, list[torch.Tensor]] = {}

    def _hook_fn(self, name: str) -> callable:
        def hook(_module: nn.Module, _input: Any, output: torch.Tensor) -> None:
            self._cache.setdefault(name, []).append(output.detach().cpu())
        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on all target streams."""
        self.remove_hooks()
        self._cache.clear()

        # Embedding streams
        self._handles.append(
            self.model.embedding.semantic.register_forward_hook(
                self._hook_fn("semantic_embed")
            )
        )
        self._handles.append(
            self.model.embedding.phase.register_forward_hook(
                self._hook_fn("phase_raw")
            )
        )
        self._handles.append(
            self.model.embedding.phase_proj.register_forward_hook(
                self._hook_fn("phase_proj")
            )
        )

        # Block-level streams
        for layer_idx, block in enumerate(self.model.blocks):
            # Pre-attention residual (input to ln1 -> attn)
            self._handles.append(
                block.ln1.register_forward_hook(
                    self._hook_fn(f"residual_pre_attn_layer{layer_idx}")
                )
            )
            # Attention output (before residual add)
            self._handles.append(
                block.attn.register_forward_hook(
                    self._hook_fn(f"attention_out_layer{layer_idx}")
                )
            )
            # Post-attention residual (after attn residual, before ff)
            # We capture this by hooking ln2 input indirectly via the ff path,
            # but a simpler approach is to hook after the attn add:
            # Since we can't easily hook the addition, we hook ln2 which sees
            # the post-attention residual stream.
            self._handles.append(
                block.ln2.register_forward_hook(
                    self._hook_fn(f"residual_post_attn_layer{layer_idx}")
                )
            )

        # Final hidden states
        self._handles.append(
            self.model.ln_final.register_forward_hook(
                self._hook_fn("final_hidden")
            )
        )

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def extract(
        self,
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass and return all captured streams.

        Args:
            input_ids: Token indices ``(batch, seq_len)``.

        Returns:
            Dictionary mapping stream names to activation tensors.
            Tensors have shape ``(batch, seq_len, dim)`` where *dim* depends
            on the stream.
        """
        self.register_hooks()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_ids)
        self.remove_hooks()

        # Each hook may fire once; unwrap single-element lists
        result: dict[str, torch.Tensor] = {}
        for name, tensors in self._cache.items():
            result[name] = tensors[0] if len(tensors) == 1 else torch.stack(tensors)
        return result

    @staticmethod
    def aggregate_streams(
        streams: dict[str, torch.Tensor],
        aggregation: str = "flatten",
    ) -> dict[str, torch.Tensor]:
        """Aggregate extracted streams for SAE training.

        Args:
            streams: Output of :meth:`extract`.
            aggregation: One of ``"flatten"`` (``batch*seq, dim``),
                ``"mean"`` (mean over sequence), or ``"last"`` (last token).

        Returns:
            Dictionary with aggregated activations.
        """
        agg: dict[str, torch.Tensor] = {}
        for name, tensor in streams.items():
            if aggregation == "flatten":
                agg[name] = tensor.view(-1, tensor.shape[-1])
            elif aggregation == "mean":
                agg[name] = tensor.mean(dim=1)
            elif aggregation == "last":
                agg[name] = tensor[:, -1, :]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        return agg


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def feature_sparsity(hidden: torch.Tensor, threshold: float = 1e-5) -> float:
    """Fraction of features active (above *threshold*) per sample.

    Args:
        hidden: Hidden activations ``(n_samples, hidden_dim)``.
        threshold: Activation threshold.

    Returns:
        Mean fraction of active features per sample.
    """
    active = (hidden > threshold).float().mean(dim=1)
    return active.mean().item()


def reconstruction_quality(
    original: torch.Tensor, reconstruction: torch.Tensor
) -> float:
    """Reconstruction quality as MSE / variance(original).

    A value << 1 indicates the reconstruction explains most of the variance.

    Args:
        original: Ground-truth activations.
        reconstruction: Reconstructed activations.

    Returns:
        Scalar quality ratio.
    """
    mse = F.mse_loss(reconstruction, original).item()
    var = original.var().item()
    if var < 1e-12:
        return 0.0
    return mse / var


def feature_entropy(hidden: torch.Tensor, eps: float = 1e-8) -> float:
    """Entropy of mean feature activations across the dataset.

    Measures how uniformly activations are distributed across features.
    High entropy means activations are spread out; low entropy means a
    few features dominate.

    Args:
        hidden: Hidden activations ``(n_samples, hidden_dim)``.
        eps: Small constant for numerical stability.

    Returns:
        Normalised entropy in ``[0, 1]``.
    """
    mean_act = hidden.mean(dim=0)
    p = mean_act / (mean_act.sum() + eps)
    p = p.clamp(min=eps)
    entropy = -(p * p.log()).sum().item()
    max_entropy = math.log(hidden.shape[1])
    if max_entropy < 1e-8:
        return 0.0
    return entropy / max_entropy


def evaluate_sae(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    batch_size: int = 1024,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Run a full evaluation of a trained SAE on a set of activations.

    Args:
        sae: The sparse autoencoder.
        activations: Input activations ``(N, input_dim)``.
        batch_size: Batch size for inference.
        device: Torch device.

    Returns:
        Dictionary of evaluation metrics.
    """
    from .device import get_device

    device = get_device(device)
    sae = sae.to(device)
    sae.eval()

    if activations.dim() == 3:
        activations = activations.view(-1, activations.shape[-1])
    activations = activations.to(device)

    all_recon: list[torch.Tensor] = []
    all_hidden: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, activations.shape[0], batch_size):
            batch = activations[i : i + batch_size]
            recon, h = sae(batch)
            all_recon.append(recon.cpu())
            all_hidden.append(h.cpu())

    recon = torch.cat(all_recon, dim=0)
    hidden = torch.cat(all_hidden, dim=0)

    return {
        "reconstruction_quality": reconstruction_quality(activations.cpu(), recon),
        "feature_sparsity": feature_sparsity(hidden),
        "feature_entropy": feature_entropy(hidden),
        "mean_activation": hidden.mean().item(),
        "dead_features": (hidden.sum(dim=0) == 0).float().mean().item(),
    }
