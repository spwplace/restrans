#!/usr/bin/env python3
"""Summarize experiment-plan output directories into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def flatten_result(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    task = data.get("task") or data.get("config", {}).get("task") or path.parent.name
    run = path.parent.name
    verdict = data.get("verdict", "")
    config = data.get("config", {})
    output_dir = str(config.get("output_dir", path.parent))
    rows = []
    for condition, metrics in data.get("summary", {}).items():
        rows.append(
            {
                "path": str(path),
                "output_dir": output_dir,
                "run": run,
                "task": task,
                "condition": condition,
                "n": metrics.get("n", ""),
                "params": metrics.get("params", ""),
                "answer_acc_mean": metrics.get("answer_acc_mean", ""),
                "answer_acc_std": metrics.get("answer_acc_std", ""),
                "acc_minus_majority": metrics.get("acc_minus_majority", ""),
                "answer_loss_gain": metrics.get("answer_loss_gain", ""),
                "label_gap": metrics.get("label_gap", ""),
                "dispersion_delta": metrics.get("dispersion_delta", ""),
                "verdict": verdict,
            }
        )
    return rows


def gate(row: dict[str, Any]) -> str:
    try:
        acc = float(row["acc_minus_majority"])
        loss = float(row["answer_loss_gain"])
        gap = float(row["label_gap"])
    except (TypeError, ValueError):
        return "unknown"
    if acc > 0.05 and loss > 0.0 and gap > 0.0:
        return "pass"
    if acc > 0.05 or loss > 0.0 or gap > 0.0:
        return "partial"
    return "fail"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gate",
        "run",
        "task",
        "condition",
        "n",
        "params",
        "answer_acc_mean",
        "answer_acc_std",
        "acc_minus_majority",
        "answer_loss_gain",
        "label_gap",
        "dispersion_delta",
        "verdict",
        "output_dir",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["gate"] = gate(row)
            writer.writerow({field: out.get(field, "") for field in fieldnames})


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Summary",
        "",
        "| gate | run | condition | acc | acc-majority | loss gain | label gap | verdict |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: (r.get("run", ""), r.get("task", ""), r.get("condition", ""))):
        lines.append(
            "| "
            + " | ".join(
                [
                    gate(row),
                    str(row.get("run", "")),
                    str(row.get("condition", "")),
                    fmt(row.get("answer_acc_mean", "")),
                    fmt(row.get("acc_minus_majority", "")),
                    fmt(row.get("answer_loss_gain", "")),
                    fmt(row.get("label_gap", "")),
                    str(row.get("verdict", "")).replace("|", "/"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("resonance/outputs/experiment_plan"))
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.root.rglob("results.json")):
        rows.extend(flatten_result(path))
    output_csv = args.output_csv or args.root / "summary.csv"
    output_md = args.output_md or args.root / "summary.md"
    write_csv(output_csv, rows)
    write_markdown(output_md, rows)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
