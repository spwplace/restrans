#!/usr/bin/env python3
"""Build plots and a lab notebook from accumulated resonance experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def iter_result_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("results.json")):
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("summary"), dict):
            continue
        config = data.get("config", {})
        run = path.parent.name
        task = data.get("task") or config.get("task") or run
        verdict = data.get("verdict", "")
        majority = data.get("label_balance", {}).get("majority_acc", "")
        for condition, metrics in data["summary"].items():
            rows.append(
                {
                    "path": str(path),
                    "family": path.relative_to(root).parts[0] if path != root else "",
                    "run": run,
                    "task": task,
                    "condition": condition,
                    "verdict": verdict,
                    "majority_acc": majority,
                    "n": metrics.get("n", ""),
                    "params": metrics.get("params", ""),
                    "acc": metrics.get("answer_acc_mean", ""),
                    "acc_std": metrics.get("answer_acc_std", ""),
                    "acc_minus_majority": metrics.get("acc_minus_majority", ""),
                    "loss_gain": metrics.get("answer_loss_gain", ""),
                    "label_gap": metrics.get("label_gap", ""),
                    "dispersion_delta": metrics.get("dispersion_delta", ""),
                }
            )
    return rows


def iter_corpus_lm_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("results.json")):
        data = load_json(path)
        if not isinstance(data, dict) or data.get("kind") != "corpus_lm":
            continue
        run = path.parent.name
        config = data.get("config", {})
        for item in data.get("runs", []):
            rows.append(
                {
                    "path": str(path),
                    "run": run,
                    "condition": item.get("condition", ""),
                    "params": item.get("params", ""),
                    "final_val_loss": item.get("final_val_loss", ""),
                    "final_val_ppl": item.get("final_val_ppl", ""),
                    "elapsed_sec": item.get("elapsed_sec", ""),
                    "train_chunks": data.get("train_chunks", ""),
                    "val_chunks": data.get("val_chunks", ""),
                    "device": config.get("device", ""),
                    "train_path": config.get("train_path", ""),
                }
            )
    return rows


def f(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def gate(row: dict[str, Any]) -> str:
    if f(row["acc_minus_majority"]) > 0.05 and f(row["loss_gain"]) > 0 and f(row["label_gap"]) > 0:
        return "pass"
    if f(row["acc_minus_majority"]) > 0.05 or f(row["loss_gain"]) > 0 or f(row["label_gap"]) > 0:
        return "partial"
    return "fail"


def fmt(value: Any) -> str:
    value_f = f(value)
    if math.isnan(value_f):
        return str(value)
    return f"{value_f:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gate",
        "family",
        "run",
        "task",
        "condition",
        "acc",
        "acc_std",
        "acc_minus_majority",
        "loss_gain",
        "label_gap",
        "dispersion_delta",
        "n",
        "params",
        "verdict",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["gate"] = gate(row)
            writer.writerow({field: out.get(field, "") for field in fieldnames})


def try_import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def best_delta_rows(rows: list[dict[str, Any]], baseline: str = "standard") -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_run[row["run"]][row["condition"]] = row
    deltas = []
    for run, conds in by_run.items():
        if baseline not in conds:
            continue
        base = conds[baseline]
        for condition, row in conds.items():
            if condition == baseline:
                continue
            out = dict(row)
            out["acc_delta_vs_standard"] = f(row["acc"]) - f(base["acc"])
            out["loss_gain_delta_vs_standard"] = f(row["loss_gain"]) - f(base["loss_gain"])
            out["label_gap_delta_vs_standard"] = f(row["label_gap"]) - f(base["label_gap"])
            deltas.append(out)
    return sorted(deltas, key=lambda r: f(r["acc_delta_vs_standard"], -999), reverse=True)


def plot_bars(plt, rows: list[dict[str, Any]], corpus_rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    deltas = [r for r in best_delta_rows(rows) if not math.isnan(f(r["acc_delta_vs_standard"]))]
    top = deltas[:30]
    if top:
        fig, ax = plt.subplots(figsize=(12, max(5, len(top) * 0.28)))
        labels = [f"{r['run']} / {r['condition']}" for r in reversed(top)]
        values = [f(r["acc_delta_vs_standard"]) for r in reversed(top)]
        ax.barh(labels, values)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Accuracy delta vs standard")
        ax.set_title("Top architecture deltas across completed probes")
        fig.tight_layout()
        path = out_dir / "top_accuracy_deltas.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    wh = [r for r in rows if r["run"].startswith("blimp_wh_island")]
    if wh:
        runs = sorted({r["run"] for r in wh})
        conditions = sorted({r["condition"] for r in wh})
        x = list(range(len(runs)))
        width = 0.8 / max(1, len(conditions))
        fig, ax = plt.subplots(figsize=(13, 6))
        for idx, condition in enumerate(conditions):
            vals = []
            for run in runs:
                match = next((r for r in wh if r["run"] == run and r["condition"] == condition), None)
                vals.append(f(match["acc"]) if match else math.nan)
            xs = [i - 0.4 + width / 2 + idx * width for i in x]
            ax.bar(xs, vals, width=width, label=condition)
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=35, ha="right")
        ax.set_ylim(0.45, 1.02)
        ax.set_ylabel("Validation accuracy")
        ax.set_title("BLiMP wh-island low-data and saturation sweep")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "blimp_wh_island_sweep.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    scatter_rows = [r for r in rows if not math.isnan(f(r["acc_minus_majority"])) and not math.isnan(f(r["label_gap"]))]
    if scatter_rows:
        fig, ax = plt.subplots(figsize=(9, 7))
        for condition, cond_rows in by_condition.items():
            xs = [f(r["acc_minus_majority"]) for r in cond_rows]
            ys = [f(r["label_gap"]) for r in cond_rows]
            ax.scatter(xs, ys, label=condition, alpha=0.72, s=28)
        ax.axvline(0.05, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Accuracy minus majority")
        ax.set_ylabel("Label geometry gap")
        ax.set_title("Regime-gate geometry across probes")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / "gate_scatter.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    if corpus_rows:
        by_run = defaultdict(list)
        for row in corpus_rows:
            by_run[row["run"]].append(row)
        for run, run_rows in by_run.items():
            fig, ax = plt.subplots(figsize=(9, 5))
            labels = [r["condition"] for r in run_rows]
            values = [f(r["final_val_ppl"]) for r in run_rows]
            ax.bar(labels, values)
            ax.set_ylabel("Validation perplexity")
            ax.set_title(f"Corpus LM PPL: {run}")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            path = out_dir / f"corpus_lm_{run}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)

    return paths


def write_notebook(
    path: Path,
    rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    plots: list[Path],
    repo_root: Path,
) -> None:
    total = len({r["run"] for r in rows})
    by_gate = defaultdict(int)
    for row in rows:
        by_gate[gate(row)] += 1
    deltas = best_delta_rows(rows)
    interesting = [
        r
        for r in deltas
        if f(r["acc_delta_vs_standard"]) > 0.03
        or (f(r["loss_gain"]) > 0 and f(r["label_gap"]) > 0 and f(r["acc_minus_majority"]) > 0.05)
    ][:20]

    lines = [
        "# Resonance Transformer Lab Notebook",
        "",
        "Generated from local experiment artifacts. This notebook is intentionally empirical: it records what ran, what moved, and what the current claims should be.",
        "",
        "## Current Read",
        "",
        "- The original strong claim has narrowed in the right direction: current evidence supports investigating a separate phase/structural channel, not yet a causal win for the resonance attention bias.",
        "- The best natural-language signal remains BLiMP `wh_island` in a low-data regime. The effect is an accuracy hint with marginal loss/geometry support, so it is a candidate regime, not a result.",
        "- Several simple agreement probes and saturated settings are useful negatives: they prevent us from mistaking easy lexical or data-regime effects for topology.",
        "- Full resonance and phase-only often track each other closely; bias-only and inert often track each other closely. That pairing is now a key diagnostic.",
        "",
        "## Artifact Inventory",
        "",
        f"- Completed run directories found: `{total}`.",
        f"- Condition rows: `{len(rows)}`.",
        f"- Corpus LM rows: `{len(corpus_rows)}`.",
        f"- Gate row counts: pass `{by_gate['pass']}`, partial `{by_gate['partial']}`, fail `{by_gate['fail']}`.",
        "",
        "## Plots",
        "",
    ]
    for plot in plots:
        rel = plot.relative_to(path.parent)
        lines.append(f"![{plot.stem}]({rel.as_posix()})")
        lines.append("")

    lines.extend(
        [
            "## Most Interesting Architecture Deltas",
            "",
            "| run | condition | acc | delta vs standard | loss gain | label gap | verdict |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in interesting:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"],
                    row["condition"],
                    fmt(row["acc"]),
                    fmt(row["acc_delta_vs_standard"]),
                    fmt(row["loss_gain"]),
                    fmt(row["label_gap"]),
                    row["verdict"].replace("|", "/"),
                ]
            )
            + " |"
        )

    if corpus_rows:
        lines.extend(
            [
                "",
                "## Corpus LM Runs",
                "",
                "These runs train on BabyLM-style natural text chunks rather than binary structural probes. They are not direct evidence for the topology claim, but they are useful for checking whether the architecture behaves sanely on a real language-modeling objective.",
                "",
                "| run | device | condition | chunks | val loss | val ppl | elapsed min |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in corpus_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["run"],
                        str(row["device"]),
                        row["condition"],
                        str(row["train_chunks"]),
                        fmt(row["final_val_loss"]),
                        fmt(row["final_val_ppl"]),
                        fmt(f(row["elapsed_sec"]) / 60),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Core Negative Controls",
            "",
            "- Saturation is common. When every condition reaches very high accuracy, the task cannot identify the useful mechanism.",
            "- Low-data undertraining is also common. Accuracy bumps without loss improvement or label-gap movement should be treated as scouting signal.",
            "- `resonance_inert_normalized` is essential. If inert matches bias-only, the relation bias is not doing meaningful relation-aware work in that regime.",
            "",
            "## Next Experimental Decisions",
            "",
            "1. Promote only regimes where phase/full beat standard and inert controls on accuracy, loss, and geometry.",
            "2. Add a cheap iso-parameter baseline for the phase path so parameter count is not confounded with the structural-channel claim.",
            "3. Search harder for tasks where full resonance separates from phase-only. Graph/semantic-parse/code-pair probes are better candidates than simple syntax.",
            "4. Use interpretability tools after regime selection: phase ablation, phase quantization, phase-feature clustering, SAE-on-phase, and attention/resonance visualizations.",
            "",
            "## Complete Summary Table",
            "",
            "| gate | family | run | condition | acc | acc-majority | loss gain | label gap | verdict |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda r: (r["family"], r["run"], r["condition"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    gate(row),
                    row["family"],
                    row["run"],
                    row["condition"],
                    fmt(row["acc"]),
                    fmt(row["acc_minus_majority"]),
                    fmt(row["loss_gain"]),
                    fmt(row["label_gap"]),
                    row["verdict"].replace("|", "/"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("resonance/outputs"))
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/lab_notebook"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = iter_result_rows(args.root)
    corpus_rows = iter_corpus_lm_rows(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_all_results.csv", rows)
    plots: list[Path] = []
    plt = try_import_pyplot()
    if plt is not None:
        plots = plot_bars(plt, rows, corpus_rows, args.output_dir / "plots")
    write_notebook(args.output_dir / "lab-notebook.md", rows, corpus_rows, plots, Path.cwd())
    print(f"Wrote {args.output_dir / 'summary_all_results.csv'}")
    print(f"Wrote {args.output_dir / 'lab-notebook.md'}")
    for plot in plots:
        print(f"Wrote {plot}")


if __name__ == "__main__":
    main()
