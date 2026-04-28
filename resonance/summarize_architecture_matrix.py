#!/usr/bin/env python3
"""Summarize architecture-matrix structural-task runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_task_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for results_path in sorted(root.glob("*/results.json")):
        try:
            raw = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            continue
        task_name = results_path.parent.name
        for condition, summary in raw.get("summary", {}).items():
            rows.append(
                {
                    "task": task_name,
                    "condition": condition,
                    "params": int(summary.get("params", 0)),
                    "answer_acc_mean": float(summary.get("answer_acc_mean", 0.0)),
                    "acc_minus_majority": float(summary.get("acc_minus_majority", 0.0)),
                    "answer_loss_gain": float(summary.get("answer_loss_gain", 0.0)),
                    "dispersion_delta": float(summary.get("dispersion_delta", 0.0)),
                    "label_gap": float(summary.get("label_gap", 0.0)),
                    "n": float(summary.get("n", 0.0)),
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    aggregates = []
    for condition, items in grouped.items():
        aggregates.append(
            {
                "condition": condition,
                "tasks_completed": len(items),
                "mean_acc": statistics.fmean(item["answer_acc_mean"] for item in items),
                "mean_acc_minus_majority": statistics.fmean(item["acc_minus_majority"] for item in items),
                "mean_loss_gain": statistics.fmean(item["answer_loss_gain"] for item in items),
                "mean_label_gap": statistics.fmean(item["label_gap"] for item in items),
                "mean_dispersion_delta": statistics.fmean(item["dispersion_delta"] for item in items),
                "max_acc": max(item["answer_acc_mean"] for item in items),
                "best_task": max(items, key=lambda item: item["answer_acc_mean"])["task"],
                "params": items[0]["params"],
            }
        )
    aggregates.sort(key=lambda item: (item["mean_acc"], item["mean_loss_gain"]), reverse=True)
    return aggregates


def describe_root(root: Path) -> tuple[str, str]:
    name = root.name
    if name.startswith("signal_matrix"):
        return (
            "Signal Matrix",
            "This run focuses compute on structure-heavy tasks that are more likely to distinguish architectural mechanisms.",
        )
    if name.startswith("literature_regime"):
        return (
            "Literature-Regime Matrix",
            "This run tests currently implemented lessons from relation-aware, dual-stream, and interpretable-training literature.",
        )
    if name.startswith("architecture_matrix"):
        return (
            "Architecture Matrix",
            "This run exercises registered architecture variants through the shared structural-task probe harness.",
        )
    return (
        "Structural Probe Matrix",
        "This run aggregates completed structural-task probes under a shared output root.",
    )


def write_markdown(path: Path, root: Path, rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    tasks = sorted({row["task"] for row in rows})
    title, description = describe_root(root)
    lines = [
        f"# {title}",
        "",
        f"Root: `{root}`",
        "",
        description,
        "",
        f"- Conditions completed: `{len(aggregates)}`",
        f"- Tasks completed: `{', '.join(tasks)}`",
        "",
        "## Aggregate Ranking",
        "",
        "| Rank | Condition | Tasks | Mean Acc | Acc-Maj | Loss Gain | Label Gap | Best Task | Params |",
        "|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for idx, row in enumerate(aggregates, start=1):
        lines.append(
            f"| {idx} | `{row['condition']}` | {row['tasks_completed']} | "
            f"{row['mean_acc']:.4f} | {row['mean_acc_minus_majority']:+.4f} | "
            f"{row['mean_loss_gain']:+.4f} | {row['mean_label_gap']:+.4f} | "
            f"{row['best_task']} | {row['params']:,} |"
        )
    lines.extend(
        [
            "",
            "## Per-Task Rows",
            "",
            "| Task | Condition | Acc | Acc-Maj | Loss Gain | Label Gap | Params |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["task"], -item["answer_acc_mean"], item["condition"])):
        lines.append(
            f"| {row['task']} | `{row['condition']}` | {row['answer_acc_mean']:.4f} | "
            f"{row['acc_minus_majority']:+.4f} | {row['answer_loss_gain']:+.4f} | "
            f"{row['label_gap']:+.4f} | {row['params']:,} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_task_rows(args.root)
    aggregates = aggregate(rows)
    payload = {
        "root": str(args.root),
        "rows": rows,
        "aggregate": aggregates,
    }
    output_json = args.output_json or args.root / "summary.json"
    output_md = args.output_md or args.root / "summary.md"
    output_json.write_text(json.dumps(payload, indent=2))
    write_markdown(output_md, args.root, rows, aggregates)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
