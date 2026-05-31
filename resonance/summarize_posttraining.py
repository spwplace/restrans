#!/usr/bin/env python3
"""Summarize structural posttraining runs.

Scans a root produced by ``scripts/run_structural_posttraining.sh`` and writes:

* ``posttraining_summary.csv``
* ``posttraining_summary.md``
* ``posttraining_accuracy.png`` when matplotlib is available
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/results.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        algorithm = path.parent.parent.name
        task = data.get("task") or path.parent.name
        summary = data.get("summary", {})
        for condition, values in summary.items():
            rows.append(
                {
                    "algorithm": algorithm,
                    "task": task,
                    "condition": condition,
                    "params": int(values.get("params", 0)),
                    "before_acc": float(values.get("before_acc_mean", 0.0)),
                    "final_acc": float(values.get("answer_acc_mean", 0.0)),
                    "acc_gain": float(values.get("acc_gain", 0.0)),
                    "acc_minus_majority": float(values.get("acc_minus_majority", 0.0)),
                    "margin": float(values.get("answer_margin_mean", 0.0)),
                    "loss": float(values.get("answer_loss_mean", 0.0)),
                    "phase_gap": float(values.get("phase_label_gap", 0.0)),
                    "hidden_gap": float(values.get("hidden_label_gap", 0.0)),
                    "path": str(path),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = [
        "algorithm",
        "task",
        "condition",
        "params",
        "before_acc",
        "final_acc",
        "acc_gain",
        "acc_minus_majority",
        "margin",
        "loss",
        "phase_gap",
        "hidden_gap",
        "path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Structural Posttraining Summary",
        "",
    ]
    if not rows:
        lines.append("No completed `results.json` files found yet.")
        path.write_text("\n".join(lines))
        return
    lines.extend(
        [
            "| Algorithm | Task | Condition | Final Acc | Gain | Acc-Majority | Margin | Phase Gap | Hidden Gap |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda r: (r["algorithm"], r["task"], -r["final_acc"], r["condition"])):
        lines.append(
            f"| {row['algorithm']} | {row['task']} | {row['condition']} | "
            f"{row['final_acc']:.4f} | {row['acc_gain']:+.4f} | "
            f"{row['acc_minus_majority']:+.4f} | {row['margin']:.4f} | "
            f"{row['phase_gap']:.4f} | {row['hidden_gap']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: prefer variants that improve final accuracy or margin while also increasing phase geometry or phase-ablation relevance. A pure accuracy win without structural evidence is only a model-capacity/training result.",
        ]
    )
    path.write_text("\n".join(lines))


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [f"{r['algorithm']}\n{r['task']}\n{r['condition']}" for r in rows]
    values = [r["final_acc"] for r in rows]
    colors = [
        "#5b6c8f" if r["condition"].startswith("standard") else "#b56b45"
        for r in rows
    ]
    width = max(10, min(42, 0.42 * len(rows)))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation verifier accuracy")
    ax.set_title("Structural Posttraining Accuracy")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.root)
    write_csv(args.root / "posttraining_summary.csv", rows)
    write_markdown(args.root / "posttraining_summary.md", rows)
    write_plot(args.root / "posttraining_accuracy.png", rows)
    print(f"Found {len(rows)} completed condition rows under {args.root}")
    print(f"Wrote {args.root / 'posttraining_summary.md'}")


if __name__ == "__main__":
    main()
