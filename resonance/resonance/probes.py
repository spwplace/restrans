"""Linear probe infrastructure for Resonance Transformer interpretability.

Probes test whether internal representations (semantic embeddings, phase
embeddings, hidden states, attention outputs) encode structural properties
such as syntax, AST roles, proof steps, or graph topology.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import ResonanceTransformer, StandardTransformer


# ---------------------------------------------------------------------------
# sklearn fallback helpers
# ---------------------------------------------------------------------------

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        r2_score,
    )

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


def _accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Classification accuracy as a float in [0, 1]."""
    return (preds == labels).float().mean().item()


def _f1_macro(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro F1 computed purely in torch (no sklearn)."""
    classes = torch.unique(labels)
    if classes.numel() == 0:
        return 0.0
    f1s: list[float] = []
    for c in classes:
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)
    return sum(f1s) / len(f1s)


def _r2(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """R² coefficient of determination in pure torch."""
    ss_res = ((labels - preds) ** 2).sum().item()
    mean_label = labels.mean().item()
    ss_tot = ((labels - mean_label) ** 2).sum().item()
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# LinearProbe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Simple linear probe for classification or regression.

    Args:
        input_dim: Dimensionality of the representation vectors.
        output_dim: Number of classes (classification) or 1 (regression).
        task: ``"classification"`` or ``"regression"``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        task: str = "classification",
    ) -> None:
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got {task}")
        self.task = task
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(..., input_dim)``.

        Returns:
            Logits (classification) or predictions (regression) of shape
            ``(..., output_dim)``.
        """
        return self.linear(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices or regression values.

        Args:
            x: Input tensor of shape ``(..., input_dim)``.

        Returns:
            Predictions of shape ``(...,)``.
        """
        with torch.no_grad():
            out = self.forward(x)
            if self.task == "classification":
                return out.argmax(dim=-1)
            return out.squeeze(-1)

    def loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the appropriate loss for the task.

        Args:
            x: Input tensor of shape ``(N, input_dim)``.
            targets: For classification, integer class indices ``(N,)``.
                For regression, float targets ``(N,)`` or ``(N, 1)``.

        Returns:
            Scalar loss tensor.
        """
        out = self.forward(x)
        if self.task == "classification":
            return F.cross_entropy(out, targets)
        # Regression
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        return F.mse_loss(out, targets)


# ---------------------------------------------------------------------------
# ProbeTrainer
# ---------------------------------------------------------------------------

class ProbeTrainer:
    """Train and evaluate :class:`LinearProbe` instances.

    Tracks accuracy / F1 (classification) or R² (regression) on a held-out
    validation split.  Also supports k-fold cross-validation.
    """

    def __init__(
        self,
        probe: LinearProbe,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | str | None = None,
    ) -> None:
        from .device import get_device

        self.probe = probe
        self.device = get_device(device)
        self.probe.to(self.device)
        self.optimizer = torch.optim.Adam(
            probe.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": [],
        }

    @staticmethod
    def _compute_metric(
        probe: LinearProbe,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        """Return loss and task-specific metric for a tensor batch."""
        with torch.no_grad():
            probe.eval()
            x = x.to(probe.linear.weight.device)
            y = y.to(probe.linear.weight.device)
            loss = probe.loss(x, y).item()
            preds = probe.predict(x)
            labels = y if probe.task == "classification" else y.squeeze()
            if probe.task == "classification":
                acc = _accuracy(preds, labels)
                f1 = _f1_macro(preds, labels)
                return {"loss": loss, "accuracy": acc, "f1": f1}
            else:
                r2 = _r2(preds, labels)
                return {"loss": loss, "r2": r2}

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int | None = None,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train the probe on labelled representations.

        Args:
            x_train: Training representations ``(N_train, input_dim)``.
            y_train: Training labels ``(N_train,)``.
            x_val: Optional validation representations.
            y_val: Optional validation labels.
            epochs: Maximum number of training epochs.
            batch_size: Batch size for SGD.
            patience: Early-stopping patience (epochs without improvement).
                ``None`` disables early stopping.
            verbose: Whether to print per-epoch metrics.

        Returns:
            Best validation metrics (or final training metrics if no val).
        """
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        if x_val is not None:
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)

        n_samples = x_train.shape[0]
        best_metric = -float("inf")
        best_state: dict[str, torch.Tensor] = {}
        patience_counter = 0

        for epoch in range(epochs):
            self.probe.train()
            perm = torch.randperm(n_samples, device=self.device)
            epoch_losses: list[float] = []

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                batch_x = x_train[idx]
                batch_y = y_train[idx]

                self.optimizer.zero_grad()
                loss = self.probe.loss(batch_x, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = sum(epoch_losses) / len(epoch_losses)
            self.history["train_loss"].append(train_loss)

            if x_val is not None:
                val_metrics = self._compute_metric(self.probe, x_val, y_val)
                val_loss = val_metrics["loss"]
                val_metric = val_metrics.get("accuracy", val_metrics.get("r2", 0.0))
                self.history["val_loss"].append(val_loss)
                self.history["val_metric"].append(val_metric)

                if val_metric > best_metric:
                    best_metric = val_metric
                    best_state = {
                        k: v.cpu().clone() for k, v in self.probe.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    task_str = (
                        f"acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}"
                        if self.probe.task == "classification"
                        else f"r2={val_metrics['r2']:.4f}"
                    )
                    print(
                        f"  Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | "
                        f"val_loss={val_loss:.4f} | {task_str}"
                    )

                if patience is not None and patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break
            else:
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"  Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f}")

        if best_state:
            self.probe.load_state_dict(best_state)
            return self._compute_metric(self.probe, x_val, y_val)  # type: ignore[arg-type]
        return {"loss": train_loss}

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        """Evaluate the trained probe on a test set.

        Args:
            x: Test representations ``(N, input_dim)``.
            y: Test labels ``(N,)``.

        Returns:
            Dictionary with loss and task-specific metrics.
        """
        return self._compute_metric(self.probe, x, y)

    def cross_validate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        k: int = 5,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """K-fold cross-validation.

        Args:
            x: All representations ``(N, input_dim)``.
            y: All labels ``(N,)``.
            k: Number of folds.
            epochs: Epochs per fold.
            batch_size: Batch size.
            patience: Early-stopping patience.
            verbose: Whether to print fold progress.

        Returns:
            Dictionary mapping metric names to lists of per-fold values.
        """
        n_samples = x.shape[0]
        fold_size = n_samples // k
        indices = torch.randperm(n_samples)

        all_metrics: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "f1": [],
            "r2": [],
        }

        for fold in range(k):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k - 1 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = torch.cat([indices[:val_start], indices[val_end:]])

            # Re-initialise probe for this fold
            self.probe._reset_parameters()
            self.probe.to(self.device)
            self.optimizer = torch.optim.Adam(
                self.probe.parameters(), lr=self.optimizer.defaults["lr"]
            )

            if verbose:
                print(f"Fold {fold + 1}/{k} …")

            self.train(
                x[train_idx],
                y[train_idx],
                x[val_idx],
                y[val_idx],
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                verbose=verbose,
            )
            fold_metrics = self.evaluate(x[val_idx], y[val_idx])
            all_metrics["loss"].append(fold_metrics["loss"])
            if self.probe.task == "classification":
                all_metrics["accuracy"].append(fold_metrics["accuracy"])
                all_metrics["f1"].append(fold_metrics["f1"])
            else:
                all_metrics["r2"].append(fold_metrics["r2"])

        return all_metrics


# ---------------------------------------------------------------------------
# RepresentationExtractor
# ---------------------------------------------------------------------------

class RepresentationExtractor:
    """Extract token-level or sequence-level representations from a model.

    Supports both hook-based extraction (for embedding/attention streams) and
    forward-pass extraction (for residual hidden states).
    """

    def __init__(
        self,
        model: ResonanceTransformer | StandardTransformer,
    ) -> None:
        self.model = model
        self._handles: list = []
        self._cache: dict[str, list[torch.Tensor]] = {}

    def _hook_fn(self, name: str) -> callable:
        def hook(_module: nn.Module, _input: Any, output: torch.Tensor) -> None:
            self._cache.setdefault(name, []).append(output.detach().cpu())
        return hook

    def _register_embedding_hooks(self) -> None:
        """Register hooks on embedding layers (Resonance only)."""
        if isinstance(self.model, ResonanceTransformer):
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

    def _register_block_hooks(self, layer_idx: int | None = None) -> None:
        """Register hooks on transformer blocks."""
        blocks = self.model.blocks
        if layer_idx is not None:
            blocks = [self.model.blocks[layer_idx]]
            idx_offset = layer_idx
        else:
            idx_offset = 0

        for i, block in enumerate(blocks):
            real_idx = idx_offset + i
            # Attention output (pre-residual-add)
            self._handles.append(
                block.attn.register_forward_hook(
                    self._hook_fn(f"attention_output_layer{real_idx}")
                )
            )
            # Residual stream after attention (input to ln2)
            self._handles.append(
                block.ln2.register_forward_hook(
                    self._hook_fn(f"residual_stream_layer{real_idx}")
                )
            )

    def _register_final_hook(self) -> None:
        self._handles.append(
            self.model.ln_final.register_forward_hook(
                self._hook_fn("final_hidden")
            )
        )

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def extract(
        self,
        input_ids: torch.Tensor,
        streams: list[str] | None = None,
        layer_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass and capture requested representation streams.

        Args:
            input_ids: Token indices ``(batch, seq_len)``.
            streams: List of stream names to extract.  Supported names:
                - ``semantic_embed``
                - ``phase_raw``
                - ``phase_proj``
                - ``residual_stream`` (all layers, or *layer_idx* if given)
                - ``attention_output`` (all layers, or *layer_idx* if given)
                - ``final_hidden``
                If ``None``, all available streams are extracted.
            layer_idx: If provided, only extract from this layer index.

        Returns:
            Dictionary mapping stream names to tensors of shape
            ``(batch, seq_len, dim)``.
        """
        self.remove_hooks()
        self._cache.clear()

        if streams is None:
            streams = [
                "semantic_embed",
                "phase_raw",
                "phase_proj",
                "residual_stream",
                "attention_output",
                "final_hidden",
            ]

        if "semantic_embed" in streams or "phase_raw" in streams or "phase_proj" in streams:
            self._register_embedding_hooks()
        if "residual_stream" in streams or "attention_output" in streams:
            self._register_block_hooks(layer_idx=layer_idx)
        if "final_hidden" in streams:
            self._register_final_hook()

        self.model.eval()
        with torch.no_grad():
            # For StandardTransformer we need return_hidden to capture hidden states
            if isinstance(self.model, StandardTransformer):
                _ = self.model(input_ids, return_hidden=True)
            else:
                _ = self.model(input_ids)

        self.remove_hooks()

        result: dict[str, torch.Tensor] = {}
        for name, tensors in self._cache.items():
            result[name] = tensors[0] if len(tensors) == 1 else torch.stack(tensors)

        # If user asked for residual_stream / attention_output without layer suffix,
        # return the last layer by default when layer_idx is None but multiple exist.
        # We keep the layer-suffixed names to avoid ambiguity.
        return result

    @staticmethod
    def aggregate(
        tensor: torch.Tensor,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """Aggregate a token-level tensor over the sequence dimension.

        Args:
            tensor: Tensor of shape ``(batch, seq_len, dim)``.
            aggregation: One of ``"mean"``, ``"first"``, ``"last"``, ``"cls"``.
                ``"cls"`` is an alias for ``"first"``.

        Returns:
            Tensor of shape ``(batch, dim)``.
        """
        if aggregation == "mean":
            return tensor.mean(dim=1)
        if aggregation in ("first", "cls"):
            return tensor[:, 0, :]
        if aggregation == "last":
            return tensor[:, -1, :]
        raise ValueError(f"Unknown aggregation: {aggregation}")

    def extract_and_aggregate(
        self,
        input_ids: torch.Tensor,
        streams: list[str] | None = None,
        layer_idx: int | None = None,
        aggregation: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Convenience method: extract then aggregate over sequence length.

        Returns:
            Dictionary mapping stream names to tensors of shape ``(batch, dim)``.
        """
        streams_dict = self.extract(input_ids, streams=streams, layer_idx=layer_idx)
        return {
            name: self.aggregate(tensor, aggregation)
            for name, tensor in streams_dict.items()
        }


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_probes(probe_results_dict: dict[str, dict[str, float]]) -> None:
    """Print a formatted table comparing probe results across streams.

    Args:
        probe_results_dict: Mapping from stream name (e.g. ``"semantic_embed"``)
            to metric dictionary (as returned by :meth:`ProbeTrainer.evaluate`).
    """
    if not probe_results_dict:
        print("No probe results to compare.")
        return

    # Determine which metrics are present
    sample = next(iter(probe_results_dict.values()))
    metric_names = [k for k in sample.keys() if k != "loss"]
    all_keys = ["stream", "loss"] + metric_names

    # Compute column widths
    widths = {k: len(k) for k in all_keys}
    for stream, metrics in probe_results_dict.items():
        widths["stream"] = max(widths["stream"], len(stream))
        widths["loss"] = max(widths["loss"], len(f"{metrics.get('loss', float('nan')):.4f}"))
        for m in metric_names:
            val = metrics.get(m, float("nan"))
            widths[m] = max(widths[m], len(f"{val:.4f}"))

    # Header
    header = " | ".join(k.ljust(widths[k]) for k in all_keys)
    print(header)
    print("-" * len(header))

    # Rows
    for stream, metrics in sorted(probe_results_dict.items()):
        row_parts = [stream.ljust(widths["stream"])]
        row_parts.append(f"{metrics.get('loss', float('nan')):.4f}".ljust(widths["loss"]))
        for m in metric_names:
            row_parts.append(f"{metrics.get(m, float('nan')):.4f}".ljust(widths[m]))
        print(" | ".join(row_parts))


def probe_purity_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int | None = None,
) -> float:
    """Measure how well a probe's predictions align with known labels.

    This is a simple *purity* score: for each predicted class, compute the
    fraction of samples in that cluster that belong to the majority true label.
    The overall purity is the weighted average across predicted clusters.

    Args:
        predictions: Predicted class indices ``(N,)``.
        labels: Ground-truth class indices ``(N,)``.
        num_classes: Number of predicted classes.  Inferred if ``None``.

    Returns:
        Purity score in ``[0, 1]`` where 1 indicates perfect alignment.
    """
    if predictions.numel() == 0:
        return 0.0
    if num_classes is None:
        num_classes = int(torch.max(predictions).item()) + 1

    total = predictions.numel()
    purity_sum = 0.0
    for c in range(num_classes):
        mask = predictions == c
        cluster_size = mask.sum().item()
        if cluster_size == 0:
            continue
        cluster_labels = labels[mask]
        # majority label frequency in this cluster
        max_count = int(cluster_labels.bincount(minlength=num_classes).max().item())
        purity_sum += max_count

    return purity_sum / total
