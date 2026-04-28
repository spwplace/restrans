#!/usr/bin/env python3
"""Run lightweight structural-stream interpretability audits on saved checkpoints.

This is not a replacement for full mechanistic interpretability.  It is the
handoff-ready audit that every promising medium-suite run should produce:

  - original validation accuracy/loss,
  - phase/intervention sensitivity where a phase table exists,
  - residual-vs-phase answer-state geometry,
  - resonance/attention trace diagnostics for one batch.

The script consumes checkpoints written by:

    resonance/structural_task_probe.py --save_models
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from regime_probe import (  # noqa: E402
    forward_answer_batch,
    label_dispersion,
    representation_dispersion,
)
from resonance.device import enable_deterministic, get_device
from resonance.interpretability import (  # noqa: E402
    effective_rank,
    phase_intervention,
    resonance_diagnostics,
    resonance_weight_scale,
    trace_resonance_forward,
    trace_standard_forward,
)
from resonance.models import ResonanceTransformer, StandardTransformer  # noqa: E402
from story_query_eval import collate_examples, encode_supervised_batch, evaluate  # noqa: E402
from story_topology_eval import build_model, jsonify  # noqa: E402
from structural_task_probe import build_datasets  # noqa: E402
from synthetic.semantic_story import StoryTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/interp_audit"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_eval_examples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--phase_noise_sigma", type=float, default=0.5)
    parser.add_argument("--phase_modes", nargs="+", default=["zero", "permute", "noise"])
    parser.add_argument("--resonance_scales", type=float, nargs="+", default=[0.0, 2.0])
    return parser.parse_args()


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _restore_args(raw: dict[str, Any], audit_args: argparse.Namespace) -> Namespace:
    restored = dict(raw)
    for key in ["output_dir", "data_path", "val_data_path"]:
        if restored.get(key) is not None:
            restored[key] = Path(restored[key])
    restored["val_examples"] = min(int(restored.get("val_examples", audit_args.max_eval_examples)), audit_args.max_eval_examples)
    restored["batch_size"] = int(audit_args.batch_size or restored.get("batch_size", 32))
    if audit_args.device is not None:
        restored["device"] = audit_args.device
    return Namespace(**restored)


def _restore_tokenizer(raw: dict[str, Any]) -> StoryTokenizer:
    tokenizer = StoryTokenizer(vocab_size=int(raw["vocab_size"]))
    tokenizer.word2id = {str(key): int(value) for key, value in raw["word2id"].items()}
    tokenizer.id2word = {value: key for key, value in tokenizer.word2id.items()}
    return tokenizer


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


@torch.no_grad()
def answer_state_geometry(
    model: torch.nn.Module,
    val_ds,
    tokenizer: StoryTokenizer,
    *,
    args: Namespace,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_examples)
    hidden_rows: list[torch.Tensor] = []
    phase_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for batch in loader:
        _loss, _logits, targets, hidden, phase = forward_answer_batch(
            model,
            tokenizer,
            batch,
            max_length=args.max_length,
            device=device,
        )
        hidden_rows.append(hidden.detach().cpu())
        target_rows.append(targets.detach().cpu())
        if phase is not None:
            phase_rows.append(phase.detach().cpu())
    hidden = torch.cat(hidden_rows, dim=0)
    targets = torch.cat(target_rows, dim=0)
    diagnostics = {
        "hidden_effective_rank": effective_rank(hidden),
        "hidden_dispersion": float(representation_dispersion(hidden).item()),
    }
    diagnostics.update({f"hidden_{key}": value for key, value in label_dispersion(hidden, targets).items()})
    if phase_rows:
        phase = torch.cat(phase_rows, dim=0)
        diagnostics.update(
            {
                "phase_effective_rank": effective_rank(phase),
                "phase_dispersion": float(representation_dispersion(phase).item()),
            }
        )
        diagnostics.update({f"phase_{key}": value for key, value in label_dispersion(phase, targets).items()})
    return diagnostics


@torch.no_grad()
def trace_one_batch(
    model: torch.nn.Module,
    val_ds,
    tokenizer: StoryTokenizer,
    *,
    args: Namespace,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(val_ds, batch_size=min(args.batch_size, 8), shuffle=False, collate_fn=collate_examples)
    batch = next(iter(loader))
    input_ids, labels, _answer_positions = encode_supervised_batch(
        tokenizer,
        batch,
        max_length=args.max_length,
        device=device,
    )
    if isinstance(model, ResonanceTransformer):
        trace = trace_resonance_forward(model, input_ids, labels=labels)
        return resonance_diagnostics(trace)
    if isinstance(model, StandardTransformer):
        trace = trace_standard_forward(model, input_ids, labels=labels)
        attention_delta = [float(layer.attention_delta.abs().mean().item()) for layer in trace.layers]
        return {
            "attention_delta_abs_mean": _mean(attention_delta),
            "token_embedding_effective_rank": effective_rank(trace.token_embeddings[0]),
        }
    return {}


def evaluate_with_interventions(
    model: torch.nn.Module,
    val_ds,
    tokenizer: StoryTokenizer,
    *,
    args: Namespace,
    audit_args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    original = evaluate(
        model,
        val_ds,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    rows: dict[str, Any] = {"original": original}
    if not isinstance(model, ResonanceTransformer):
        return rows
    for mode in audit_args.phase_modes:
        with phase_intervention(
            model,
            mode=mode,
            sigma=audit_args.phase_noise_sigma,
            seed=int(args.dataset_seed) + 17,
        ):
            metrics = evaluate(
                model,
                val_ds,
                tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
            )
        rows[f"phase_{mode}"] = {
            **metrics,
            "answer_acc_delta": metrics["answer_acc"] - original["answer_acc"],
            "answer_loss_delta": metrics["answer_loss"] - original["answer_loss"],
        }
    if getattr(model.config, "use_resonance_bias", False):
        for scale in audit_args.resonance_scales:
            with resonance_weight_scale(model, scale):
                metrics = evaluate(
                    model,
                    val_ds,
                    tokenizer,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=device,
                )
            rows[f"resonance_scale_{scale:g}"] = {
                **metrics,
                "answer_acc_delta": metrics["answer_acc"] - original["answer_acc"],
                "answer_loss_delta": metrics["answer_loss"] - original["answer_loss"],
            }
    return rows


def audit_checkpoint(path: Path, audit_args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    ckpt = _torch_load(path, device)
    run_args = _restore_args(ckpt["args"], audit_args)
    tokenizer = _restore_tokenizer(ckpt["tokenizer"])
    _train_ds, val_ds = build_datasets(run_args)
    model, _config = build_model(run_args, ckpt["model_condition"], tokenizer)
    model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    interventions = evaluate_with_interventions(
        model,
        val_ds,
        tokenizer,
        args=run_args,
        audit_args=audit_args,
        device=device,
    )
    geometry = answer_state_geometry(model, val_ds, tokenizer, args=run_args, device=device)
    trace = trace_one_batch(model, val_ds, tokenizer, args=run_args, device=device)
    return {
        "checkpoint": str(path),
        "task": getattr(run_args, "task", "regime_probe"),
        "condition": ckpt["condition"],
        "model_condition": ckpt["model_condition"],
        "seed": ckpt["seed"],
        "interventions": interventions,
        "geometry": geometry,
        "trace": trace,
    }


def write_report(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Interpretability Audit",
        "",
        "| Checkpoint | Task | Condition | Seed | Acc | Zero Δ | Permute Δ | Noise Δ | Hidden Rank | Phase Rank |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        interventions = row["interventions"]
        original = interventions["original"]
        geometry = row["geometry"]

        def delta(name: str) -> str:
            value = interventions.get(name, {}).get("answer_acc_delta")
            return "" if value is None or not math.isfinite(float(value)) else f"{float(value):+.4f}"

        lines.append(
            f"| {Path(row['checkpoint']).name} | {row['task']} | {row['condition']} | {row['seed']} | "
            f"{original['answer_acc']:.4f} | {delta('phase_zero')} | {delta('phase_permute')} | "
            f"{delta('phase_noise')} | {geometry.get('hidden_effective_rank', 0.0):.2f} | "
            f"{geometry.get('phase_effective_rank', 0.0):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    enable_deterministic(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    results = [audit_checkpoint(path, args, device) for path in args.checkpoints]
    (args.output_dir / "results.json").write_text(json.dumps(jsonify(results), indent=2))
    write_report(args.output_dir / "report.md", results)
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
