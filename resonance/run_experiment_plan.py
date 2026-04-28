#!/usr/bin/env python3
"""Run the first executable wave from experiment-plan.md.

This is intentionally a plain subprocess runner.  It keeps the research plan
editable as commands, emits a manifest, supports dry-runs, and avoids hiding
individual experiment failures behind a scheduler abstraction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SIZE_PRESETS = {
    "smoke": {
        "epochs": 1,
        "train_examples": 128,
        "val_examples": 64,
        "embed_dim": 64,
        "layers": 2,
        "heads": 4,
        "ff_dim": 256,
        "batch_size": 16,
    },
    "small": {
        "epochs": 5,
        "train_examples": 5_000,
        "val_examples": 1_000,
        "embed_dim": 128,
        "layers": 4,
        "heads": 4,
        "ff_dim": 512,
        "batch_size": 32,
    },
    "ten_m": {
        "epochs": 10,
        "train_examples": 20_000,
        "val_examples": 4_000,
        "embed_dim": 352,
        "layers": 6,
        "heads": 8,
        "ff_dim": 1408,
        "batch_size": 48,
    },
}


@dataclass
class PlannedRun:
    name: str
    tier: str
    hypothesis: str
    command: list[str]
    required_paths: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=["preflight", "primary", "synthetic", "story", "second_wave", "all"],
        default="primary",
    )
    parser.add_argument("--size", choices=sorted(SIZE_PRESETS), default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_root", type=Path, default=Path("resonance/outputs/experiment_plan"))
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--conditions", default="standard,resonance_full_normalized,bias_only_normalized")
    parser.add_argument("--story_conditions", default="standard,resonance_full_normalized,bias_only_normalized,resonance_inert_normalized")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--strict_paths", action="store_true")
    return parser.parse_args()


def add_common_structural_args(
    cmd: list[str],
    args: argparse.Namespace,
    preset: dict[str, int],
    overrides: dict[str, int] | None = None,
) -> list[str]:
    merged = dict(preset)
    if overrides:
        merged.update(overrides)
    cmd.extend(
        [
            "--device",
            args.device,
            "--conditions",
            args.conditions,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--epochs",
            str(merged["epochs"]),
            "--train_examples",
            str(merged["train_examples"]),
            "--val_examples",
            str(merged["val_examples"]),
            "--embed_dim",
            str(merged["embed_dim"]),
            "--layers",
            str(merged["layers"]),
            "--heads",
            str(merged["heads"]),
            "--ff_dim",
            str(merged["ff_dim"]),
            "--batch_size",
            str(merged["batch_size"]),
            "--eval_each_epoch",
        ]
    )
    return cmd


def structural_run(
    args: argparse.Namespace,
    preset: dict[str, int],
    *,
    name: str,
    tier: str,
    task: str,
    hypothesis: str,
    required_paths: list[Path] | None = None,
    extra: list[str] | None = None,
    overrides: dict[str, int] | None = None,
) -> PlannedRun:
    output_dir = args.output_root / args.size / name
    cmd = [
        args.python,
        "resonance/structural_task_probe.py",
        "--task",
        task,
        "--output_dir",
        str(output_dir),
    ]
    add_common_structural_args(cmd, args, preset, overrides)
    if extra:
        cmd.extend(extra)
    return PlannedRun(
        name=name,
        tier=tier,
        hypothesis=hypothesis,
        command=cmd,
        required_paths=[str(path) for path in required_paths or []],
    )


def story_run(args: argparse.Namespace, preset: dict[str, int]) -> PlannedRun:
    output_dir = args.output_root / args.size / "story_topology_interventions"
    cmd = [
        args.python,
        "resonance/story_topology_eval.py",
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--conditions",
        args.story_conditions,
        "--seeds",
        *[str(seed) for seed in args.seeds[:2]],
        "--epochs",
        str(preset["epochs"]),
        "--train_graphs",
        "512" if args.size == "ten_m" else "128",
        "--val_graphs",
        "128" if args.size == "ten_m" else "64",
        "--eval_graphs",
        "96" if args.size == "ten_m" else "48",
        "--embed_dim",
        str(preset["embed_dim"]),
        "--layers",
        str(preset["layers"]),
        "--heads",
        str(preset["heads"]),
        "--ff_dim",
        str(preset["ff_dim"]),
        "--batch_size",
        str(max(4, preset["batch_size"] // 4)),
        "--interventions",
    ]
    return PlannedRun(
        name="story_topology_interventions",
        tier="story",
        hypothesis=(
            "If resonance learns graph-level structure rather than lexical overlap, same-graph retrieval "
            "and hard-negative separation should improve, and phase/resonance interventions should be causal."
        ),
        command=cmd,
        required_paths=[],
    )


def build_plan(args: argparse.Namespace) -> list[PlannedRun]:
    preset = SIZE_PRESETS[args.size]
    data = args.data_dir
    probes = data / "processed/probes"
    plan: list[PlannedRun] = []

    linzen_simple = data / "processed/rnn_agreement_simple/rnn_agr_simple/numpred.train"
    linzen_val = data / "processed/rnn_agreement_simple/rnn_agr_simple/numpred.val"
    linzen_tsv = data / "processed/rnn_agreement/agr_50_mostcommon_10K.tsv"
    blimp = data / "processed/blimp"

    plan.extend(
        [
            structural_run(
                args,
                preset,
                name="linzen_simple_agreement",
                tier="primary",
                task="agreement_file",
                required_paths=[linzen_simple, linzen_val],
                hypothesis=(
                    "Subject-verb number across natural prefixes should reward a structural channel that "
                    "tracks agreement target/attractor roles rather than raw lexical continuation."
                ),
                extra=["--data_path", str(linzen_simple), "--val_data_path", str(linzen_val), "--max_length", "192"],
            ),
            structural_run(
                args,
                preset,
                name="linzen_dependency_tsv",
                tier="primary",
                task="agreement_file",
                required_paths=[linzen_tsv],
                hypothesis=(
                    "The larger Linzen dependency TSV should expose whether the effect survives broader "
                    "prefix diversity and dependency metadata."
                ),
                extra=["--data_path", str(linzen_tsv), "--max_length", "224"],
            ),
            structural_run(
                args,
                preset,
                name="blimp_distractor_agreement_relative_clause",
                tier="primary",
                task="blimp",
                required_paths=[blimp / "distractor_agreement_relative_clause.jsonl"],
                hypothesis=(
                    "Agreement with a relative-clause distractor is a direct syntax stress test; resonance "
                    "should help only if it preserves dependency geometry."
                ),
                extra=[
                    "--data_path",
                    str(blimp / "distractor_agreement_relative_clause.jsonl"),
                    "--max_length",
                    "192",
                ],
                overrides={"train_examples": 700, "val_examples": 300},
            ),
            structural_run(
                args,
                preset,
                name="blimp_anaphor_number_agreement",
                tier="primary",
                task="blimp",
                required_paths=[blimp / "anaphor_number_agreement.jsonl"],
                hypothesis=(
                    "Anaphor agreement tests binding-like structure rather than continuation fluency; "
                    "a real phase stream should show cleaner label geometry."
                ),
                extra=[
                    "--data_path",
                    str(blimp / "anaphor_number_agreement.jsonl"),
                    "--max_length",
                    "192",
                ],
                overrides={"train_examples": 700, "val_examples": 300},
            ),
            structural_run(
                args,
                preset,
                name="blimp_wh_island",
                tier="primary",
                task="blimp",
                required_paths=[blimp / "wh_island.jsonl"],
                hypothesis=(
                    "Island constraints are a harder structural grammaticality case; failure here is fine, "
                    "but a positive signal would be high-value."
                ),
                extra=[
                    "--data_path",
                    str(blimp / "wh_island.jsonl"),
                    "--max_length",
                    "224",
                ],
                overrides={"train_examples": 700, "val_examples": 300},
            ),
        ]
    )

    plan.extend(
        [
            structural_run(
                args,
                preset,
                name="graph_alias_boundary",
                tier="synthetic",
                task="graph_alias",
                hypothesis=(
                    "Aliased graph walks are the current best synthetic boundary: lexical identity is "
                    "controlled, and phase/resonance should help only by representing latent graph state."
                ),
                extra=[
                    "--n_graphs",
                    "32",
                    "--n_nodes",
                    "8",
                    "--alias_pool_size",
                    "96",
                    "--out_degree",
                    "2",
                    "--walk_length",
                    "4",
                    "--query_steps",
                    "1",
                    "--val_graphs",
                    "new",
                    "--max_length",
                    "256",
                ],
            ),
            structural_run(
                args,
                preset,
                name="template_equivalence_zero_overlap",
                tier="synthetic",
                task="template_equivalence",
                hypothesis=(
                    "Template equivalence is a pure structure indicator; use it to check whether the "
                    "architecture can ignore surface vocabulary when template identity is latent."
                ),
                extra=["--n_templates", "80", "--depth", "5", "--max_length", "256"],
            ),
            structural_run(
                args,
                preset,
                name="unification_depth4",
                tier="synthetic",
                task="unification",
                hypothesis=(
                    "First-order unification stresses variable binding and tree alignment; resonance should "
                    "help only if the phase path learns structural equivalence under renaming."
                ),
                extra=["--depth", "4", "--max_length", "256"],
            ),
            structural_run(
                args,
                preset,
                name="dyck_cross_serial",
                tier="synthetic",
                task="dyck",
                hypothesis=(
                    "Cross-serial bracket dependencies test nonlocal matching. This is a debugging probe, "
                    "not a headline result."
                ),
                extra=["--dyck_mode", "cross", "--depth", "8", "--n_types", "3", "--max_length", "192"],
            ),
            structural_run(
                args,
                preset,
                name="causal_intervention",
                tier="synthetic",
                task="causal_intervention",
                hypothesis=(
                    "Causal intervention questions should punish temporal-order heuristics and reward "
                    "edge-sensitive story topology."
                ),
                extra=["--n_events", "8", "--edge_prob", "0.30", "--max_length", "256"],
            ),
        ]
    )

    plan.append(story_run(args, preset))
    plan.extend(
        [
            structural_run(
                args,
                preset,
                name="hans_evaluation",
                tier="second_wave",
                task="generic_probe",
                required_paths=[probes / "hans/evaluation.jsonl"],
                hypothesis=(
                    "HANS tests whether the model resists lexical-overlap, subsequence, and constituent "
                    "heuristics in entailment judgments."
                ),
                extra=[
                    "--data_path",
                    str(probes / "hans/train.jsonl"),
                    "--val_data_path",
                    str(probes / "hans/evaluation.jsonl"),
                    "--max_length",
                    "256",
                ],
                overrides={"train_examples": 10_000, "val_examples": 10_000},
            ),
            structural_run(
                args,
                preset,
                name="msgs_main_verb_length",
                tier="second_wave",
                task="generic_probe",
                required_paths=[
                    probes / "msgs/main_verb_control/train.jsonl",
                    probes / "msgs/main_verb_length/test.jsonl",
                ],
                hypothesis=(
                    "MSGs checks whether linguistic structure generalizes when a surface correlate, here "
                    "length, is decorrelated at test time."
                ),
                extra=[
                    "--data_path",
                    str(probes / "msgs/main_verb_control/train.jsonl"),
                    "--val_data_path",
                    str(probes / "msgs/main_verb_length/test.jsonl"),
                    "--max_length",
                    "256",
                ],
            ),
            structural_run(
                args,
                preset,
                name="msgs_syntactic_category_relative_position",
                tier="second_wave",
                task="generic_probe",
                required_paths=[
                    probes / "msgs/syntactic_category_control/train.jsonl",
                    probes / "msgs/syntactic_category_relative_position/test.jsonl",
                ],
                hypothesis=(
                    "This MSGs split tests syntactic-category structure against a relative-position shortcut."
                ),
                extra=[
                    "--data_path",
                    str(probes / "msgs/syntactic_category_control/train.jsonl"),
                    "--val_data_path",
                    str(probes / "msgs/syntactic_category_relative_position/test.jsonl"),
                    "--max_length",
                    "256",
                ],
            ),
            structural_run(
                args,
                preset,
                name="cogs_gen_semantic_parse_probe",
                tier="second_wave",
                task="generic_probe",
                required_paths=[probes / "cogs/train.jsonl", probes / "cogs/gen.jsonl"],
                hypothesis=(
                    "COGS gen split tests whether semantic-form matching transfers compositionally rather "
                    "than memorizing training domains."
                ),
                extra=[
                    "--data_path",
                    str(probes / "cogs/train.jsonl"),
                    "--val_data_path",
                    str(probes / "cogs/gen.jsonl"),
                    "--max_length",
                    "384",
                ],
            ),
            structural_run(
                args,
                preset,
                name="slog_generalization_semantic_parse_probe",
                tier="second_wave",
                task="generic_probe",
                required_paths=[
                    probes / "slog/slog_cogs_lf_train.jsonl",
                    probes / "slog/slog_gen_cogs_lf.jsonl",
                ],
                hypothesis=(
                    "SLOG semantic-form matching stresses structural long-distance generalization under "
                    "held-out generation templates."
                ),
                extra=[
                    "--data_path",
                    str(probes / "slog/slog_cogs_lf_train.jsonl"),
                    "--val_data_path",
                    str(probes / "slog/slog_gen_cogs_lf.jsonl"),
                    "--max_length",
                    "384",
                ],
            ),
            structural_run(
                args,
                preset,
                name="cfq_mcd1_query_probe",
                tier="second_wave",
                task="generic_probe",
                required_paths=[probes / "cfq/mcd1/train.jsonl", probes / "cfq/mcd1/test.jsonl"],
                hypothesis=(
                    "CFQ MCD1 probes compositional semantic query matching under compound divergence."
                ),
                extra=[
                    "--data_path",
                    str(probes / "cfq/mcd1/train.jsonl"),
                    "--val_data_path",
                    str(probes / "cfq/mcd1/test.jsonl"),
                    "--max_length",
                    "512",
                ],
            ),
            structural_run(
                args,
                preset,
                name="poj104_problem_id_probe",
                tier="second_wave",
                task="generic_probe",
                required_paths=[probes / "poj104/train.jsonl", probes / "poj104/validation.jsonl"],
                hypothesis=(
                    "POJ-104 tests whether program structure carries enough information to identify the "
                    "underlying problem class beyond surface tokens."
                ),
                extra=[
                    "--data_path",
                    str(probes / "poj104/train.jsonl"),
                    "--val_data_path",
                    str(probes / "poj104/validation.jsonl"),
                    "--max_length",
                    "768",
                ],
                overrides={"batch_size": max(4, preset["batch_size"] // 2)},
            ),
            structural_run(
                args,
                preset,
                name="bigclonebench_clone_probe",
                tier="second_wave",
                task="generic_probe",
                required_paths=[probes / "bigclonebench/train.jsonl", probes / "bigclonebench/validation.jsonl"],
                hypothesis=(
                    "BigCloneBench is the direct program semantic-similarity test. The first generic probe "
                    "is coarse, but useful for detecting whether code pair structure is learnable."
                ),
                extra=[
                    "--data_path",
                    str(probes / "bigclonebench/train.jsonl"),
                    "--val_data_path",
                    str(probes / "bigclonebench/validation.jsonl"),
                    "--max_length",
                    "1024",
                ],
                overrides={"batch_size": max(2, preset["batch_size"] // 4)},
            ),
        ]
    )
    return [run for run in plan if args.tier == "all" or run.tier == args.tier]


def run_one(run: PlannedRun, *, dry_run: bool) -> int:
    print("\n" + "=" * 80, flush=True)
    print(f"{run.name} [{run.tier}]", flush=True)
    print(run.hypothesis, flush=True)
    print(" ".join(run.command), flush=True)
    print("=" * 80, flush=True)
    if dry_run:
        return 0
    started = time.time()
    result = subprocess.run(run.command)
    elapsed = time.time() - started
    print(f"{run.name}: exit={result.returncode} elapsed_sec={elapsed:.1f}", flush=True)
    return result.returncode


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    manifest_path = args.output_root / f"{args.tier}_{args.size}_manifest.json"
    manifest_path.write_text(json.dumps([asdict(run) for run in plan], indent=2))
    print(f"Wrote {manifest_path}")

    failures: list[tuple[str, int]] = []
    for run in plan:
        missing = [path for path in run.required_paths if not Path(path).exists()]
        if missing:
            message = f"{run.name}: missing required paths: {missing}"
            if args.strict_paths:
                raise FileNotFoundError(message)
            print(f"SKIP {message}", flush=True)
            continue
        code = run_one(run, dry_run=args.dry_run)
        if code != 0:
            failures.append((run.name, code))
            if not args.continue_on_error:
                break
    if failures:
        print(f"Failures: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
