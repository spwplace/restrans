#!/usr/bin/env python3
"""Plot a task-by-condition matrix from structural probe outputs.

The probe scripts write one ``results.json`` per task.  This utility turns a
run root into a compact CSV, a markdown report, and a few plots that are useful
for deciding whether a variant deserves a larger run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def f(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any) -> str:
    value_f = f(value)
    if math.isnan(value_f):
        return str(value)
    return f"{value_f:.4f}"


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/results.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        summary = data.get("summary")
        if not isinstance(summary, dict):
            continue
        task_dir = path.parent.name
        task = data.get("task") or data.get("config", {}).get("task") or task_dir
        majority = data.get("label_balance", {}).get("majority_acc", "")
        verdict = data.get("verdict", "")
        for condition, metrics in summary.items():
            rows.append(
                {
                    "task_dir": task_dir,
                    "task": task,
                    "condition": condition,
                    "acc": metrics.get("answer_acc_mean", ""),
                    "acc_std": metrics.get("answer_acc_std", ""),
                    "acc_minus_majority": metrics.get("acc_minus_majority", ""),
                    "loss_gain": metrics.get("answer_loss_gain", ""),
                    "label_gap": metrics.get("label_gap", ""),
                    "dispersion_delta": metrics.get("dispersion_delta", ""),
                    "params": metrics.get("params", ""),
                    "n": metrics.get("n", ""),
                    "majority_acc": majority,
                    "verdict": verdict,
                    "path": str(path),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_dir",
        "task",
        "condition",
        "acc",
        "acc_std",
        "acc_minus_majority",
        "loss_gain",
        "label_gap",
        "dispersion_delta",
        "params",
        "n",
        "majority_acc",
        "verdict",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def condition_order(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "standard",
        "standard_alibi",
        "standard_deberta_lite",
        "standard_iso_deberta_lite",
        "resonance_full_normalized",
        "phase_stream_only_normalized",
        "phase_dynamic_qk_film_normalized",
        "relation_value_mix_normalized",
        "relation_value_qk_film_normalized",
        "weighted_cosine_normalized",
        "harmonic_cosine_normalized",
        "harmonic_relation_value_normalized",
        "rbf_kernel_normalized",
        "attention_kernel_normalized",
        "relational_stream_lite_normalized",
    ]
    seen = {row["condition"] for row in rows}
    ordered = [condition for condition in preferred if condition in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def task_order(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({row["task_dir"] for row in rows})


def matrix(rows: list[dict[str, Any]], value_key: str) -> tuple[list[str], list[str], list[list[float]]]:
    tasks = task_order(rows)
    conditions = condition_order(rows)
    by_key = {(row["task_dir"], row["condition"]): row for row in rows}
    values: list[list[float]] = []
    for task in tasks:
        row_values = []
        for condition in conditions:
            row = by_key.get((task, condition))
            row_values.append(f(row.get(value_key)) if row else math.nan)
        values.append(row_values)
    return tasks, conditions, values


def delta_vs_standard(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[list[float]]]:
    tasks = task_order(rows)
    conditions = [c for c in condition_order(rows) if c != "standard"]
    by_key = {(row["task_dir"], row["condition"]): row for row in rows}
    values: list[list[float]] = []
    for task in tasks:
        base = by_key.get((task, "standard"))
        base_acc = f(base.get("acc")) if base else math.nan
        row_values = []
        for condition in conditions:
            row = by_key.get((task, condition))
            row_values.append(f(row.get("acc")) - base_acc if row and not math.isnan(base_acc) else math.nan)
        values.append(row_values)
    return tasks, conditions, values


def try_import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        return plt, np
    except Exception:
        return None, None


def plot_heatmap(plt, np, tasks: list[str], conditions: list[str], values: list[list[float]], path: Path, title: str, cmap: str) -> None:
    if not tasks or not conditions:
        return
    arr = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(max(9, len(conditions) * 0.68), max(4.5, len(tasks) * 0.48)))
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=9)
    ax.set_title(title)
    for i in range(len(tasks)):
        for j in range(len(conditions)):
            value = arr[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scatter(plt, rows: list[dict[str, Any]], path: Path) -> None:
    scatter_rows = [
        row
        for row in rows
        if not math.isnan(f(row["acc_minus_majority"])) and not math.isnan(f(row["label_gap"]))
    ]
    if not scatter_rows:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    conditions = condition_order(scatter_rows)
    for condition in conditions:
        cond_rows = [row for row in scatter_rows if row["condition"] == condition]
        ax.scatter(
            [f(row["acc_minus_majority"]) for row in cond_rows],
            [f(row["label_gap"]) for row in cond_rows],
            label=condition,
            s=34,
            alpha=0.75,
        )
    ax.axvline(0.05, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Accuracy minus majority")
    ax.set_ylabel("Label geometry gap")
    ax.set_title("Regime gate: behavior vs geometry")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path: Path, root: Path, rows: list[dict[str, Any]], plot_paths: list[Path]) -> None:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_dir"], []).append(row)

    lines = [
        "# Matrix Result Report",
        "",
        f"Root: `{root}`",
        "",
        f"- Tasks completed: `{len(by_task)}`",
        f"- Condition rows: `{len(rows)}`",
        "",
        "## Plots",
        "",
    ]
    for plot_path in plot_paths:
        rel = plot_path.relative_to(path.parent)
        lines.append(f"![{plot_path.stem}]({rel.as_posix()})")
        lines.append("")

    lines.extend(
        [
            "## Winners By Task",
            "",
            "| Task | Best Acc Condition | Acc | Delta vs Standard | Best Loss-Gain Condition | Loss Gain | Verdict |",
            "|---|---|---:|---:|---|---:|---|",
        ]
    )
    for task, task_rows in sorted(by_task.items()):
        best_acc = max(task_rows, key=lambda row: f(row["acc"], -999))
        best_loss = max(task_rows, key=lambda row: f(row["loss_gain"], -999))
        base = next((row for row in task_rows if row["condition"] == "standard"), None)
        delta = f(best_acc["acc"]) - f(base["acc"]) if base else math.nan
        lines.append(
            "| "
            + " | ".join(
                [
                    task,
                    best_acc["condition"],
                    fmt(best_acc["acc"]),
                    fmt(delta),
                    best_loss["condition"],
                    fmt(best_loss["loss_gain"]),
                    best_acc["verdict"].replace("|", "/"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Full Rows",
            "",
            "| Task | Condition | Acc | Acc-Maj | Loss Gain | Label Gap | Params |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda r: (r["task_dir"], r["condition"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task_dir"],
                    row["condition"],
                    fmt(row["acc"]),
                    fmt(row["acc_minus_majority"]),
                    fmt(row["loss_gain"]),
                    fmt(row["label_gap"]),
                    fmt(row["params"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.root / "matrix_plots"
    rows = load_rows(args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "matrix_rows.csv", rows)

    plot_paths: list[Path] = []
    plt, np = try_import_pyplot()
    if plt is not None and np is not None and rows:
        tasks, conditions, values = matrix(rows, "acc")
        path = out_dir / "accuracy_heatmap.png"
        plot_heatmap(plt, np, tasks, conditions, values, path, "Accuracy by task and condition", "viridis")
        plot_paths.append(path)

        tasks, conditions, values = delta_vs_standard(rows)
        path = out_dir / "delta_vs_standard_heatmap.png"
        plot_heatmap(plt, np, tasks, conditions, values, path, "Accuracy delta vs standard", "coolwarm")
        plot_paths.append(path)

        path = out_dir / "gate_scatter.png"
        plot_scatter(plt, rows, path)
        plot_paths.append(path)

    write_report(out_dir / "matrix-report.md", args.root, rows, plot_paths)
    print(f"Wrote {out_dir / 'matrix_rows.csv'}")
    print(f"Wrote {out_dir / 'matrix-report.md'}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
