#!/usr/bin/env python3
"""Run the shared regime probe on structural tasks.

This is the switchboard for the current research direction: use literature or
literature-like tasks as primary experiments, and keep microvalidations as
indicators.  Every task emits examples with ``prefix``, ``answer`` and
``full_text``, so the same answer-token training loop and geometry diagnostics
apply across syntax, graph, unification, bracket, and semantic-story probes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from regime_probe import (  # noqa: E402
    jsonify,
    label_balance,
    regime_verdict,
    summarise,
    train_condition,
    write_report,
)
from resonance.device import enable_deterministic  # noqa: E402
from story_query_eval import TemporalQueryDataset  # noqa: E402
from synthetic.agreement import AgreementDataset  # noqa: E402
from synthetic.causal_intervention import CausalInterventionDataset  # noqa: E402
from synthetic.dyck import DyckDataset  # noqa: E402
from synthetic.external_syntax import AgreementFileDataset, BlimpMinimalPairDataset, GenericProbeDataset  # noqa: E402
from synthetic.graph_alias import GraphAliasDataset  # noqa: E402
from synthetic.semantic_story import StoryTokenizer  # noqa: E402
from synthetic.structural_paraphrase import StructuralParaphraseDataset  # noqa: E402
from synthetic.template_equivalence import TemplateEquivalenceDataset  # noqa: E402
from synthetic.unification import UnificationDataset  # noqa: E402


TASKS = (
    "agreement",
    "agreement_file",
    "blimp",
    "generic_probe",
    "template_equivalence",
    "dyck",
    "unification",
    "graph_alias",
    "causal_intervention",
    "structural_paraphrase",
    "temporal_query",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="agreement")
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/structural_task_probe"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", default="standard,resonance_full_normalized,bias_only_normalized")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_examples", type=int, default=1024)
    parser.add_argument("--val_examples", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--phase_init_std", type=float, default=0.3)
    parser.add_argument("--story_phase_prior_std", type=float, default=1.2)
    parser.add_argument("--story_phase_prior_noise", type=float, default=0.05)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--dataset_seed", type=int, default=9000)
    parser.add_argument("--dispersion_lambda", type=float, default=0.0)
    parser.add_argument("--eval_each_epoch", action="store_true")

    # Shared task shape controls.
    parser.add_argument("--max_attractors", type=int, default=4)
    parser.add_argument("--force_attractors", type=int, default=None)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n_templates", type=int, default=50)
    parser.add_argument("--dyck_mode", choices=["nested", "cross"], default="nested")
    parser.add_argument("--n_types", type=int, default=3)
    parser.add_argument("--n_events", type=int, default=6)
    parser.add_argument("--edge_prob", type=float, default=0.35)

    # Graph alias controls.
    parser.add_argument("--n_graphs", type=int, default=16)
    parser.add_argument("--n_nodes", type=int, default=16)
    parser.add_argument("--aliases_per_node", type=int, default=3)
    parser.add_argument("--alias_pool_size", type=int, default=0)
    parser.add_argument("--out_degree", type=int, default=2)
    parser.add_argument("--walk_length", type=int, default=4)
    parser.add_argument("--query_steps", type=int, default=1)
    parser.add_argument("--val_graphs", choices=["same", "same_aliases_new_edges", "new"], default="new")

    # External local files.
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--val_data_path", type=Path, default=None)
    return parser.parse_args()


def build_graph_alias(args: argparse.Namespace) -> tuple[GraphAliasDataset, GraphAliasDataset]:
    common = {
        "n_graphs": args.n_graphs,
        "n_nodes": args.n_nodes,
        "aliases_per_node": args.aliases_per_node,
        "alias_pool_size": args.alias_pool_size if args.alias_pool_size > 0 else None,
        "alias_pool_seed": args.dataset_seed + 300_000,
        "out_degree": args.out_degree,
        "walk_length": args.walk_length,
        "query_steps": args.query_steps,
    }
    train_ds = GraphAliasDataset(n_examples=args.train_examples, seed=args.dataset_seed, **common)
    val_graph_seed = args.dataset_seed if args.val_graphs != "new" else args.dataset_seed + 100_000
    val_edge_seed = args.dataset_seed + 200_000 if args.val_graphs == "same_aliases_new_edges" else None
    val_ds = GraphAliasDataset(
        n_examples=args.val_examples,
        seed=args.dataset_seed + 100_000,
        graph_seed=val_graph_seed,
        edge_seed=val_edge_seed,
        example_seed=args.dataset_seed + 100_000,
        **common,
    )
    return train_ds, val_ds


def build_external(args: argparse.Namespace):
    if args.data_path is None:
        raise ValueError(f"--data_path is required for task {args.task}")
    val_path = args.val_data_path or args.data_path
    if args.task == "generic_probe":
        train_ds = GenericProbeDataset(path=args.data_path, n_examples=args.train_examples, offset=0)
        val_offset = 0 if args.val_data_path else args.train_examples
        val_ds = GenericProbeDataset(path=val_path, n_examples=args.val_examples, offset=val_offset)
    elif args.task == "blimp":
        train_ds = BlimpMinimalPairDataset(path=args.data_path, n_examples=args.train_examples, offset=0)
        val_offset = 0 if args.val_data_path else args.train_examples
        val_ds = BlimpMinimalPairDataset(path=val_path, n_examples=args.val_examples, offset=val_offset)
    else:
        train_ds = AgreementFileDataset(path=args.data_path, n_examples=args.train_examples, offset=0)
        val_offset = 0 if args.val_data_path else args.train_examples
        val_ds = AgreementFileDataset(path=val_path, n_examples=args.val_examples, offset=val_offset)
    return train_ds, val_ds


def build_datasets(args: argparse.Namespace):
    if args.task == "agreement":
        common = {
            "max_attractors": args.max_attractors,
            "force_attractors": args.force_attractors,
        }
        return (
            AgreementDataset(n_examples=args.train_examples, seed=args.dataset_seed, **common),
            AgreementDataset(n_examples=args.val_examples, seed=args.dataset_seed + 100_000, **common),
        )
    if args.task in {"agreement_file", "blimp", "generic_probe"}:
        return build_external(args)
    if args.task == "template_equivalence":
        common = {"n_templates": args.n_templates, "depth": args.depth}
        return (
            TemplateEquivalenceDataset(n_examples=args.train_examples, seed=args.dataset_seed, **common),
            TemplateEquivalenceDataset(n_examples=args.val_examples, seed=args.dataset_seed + 100_000, **common),
        )
    if args.task == "dyck":
        common = {"max_depth": args.depth, "n_types": args.n_types, "mode": args.dyck_mode}
        return (
            DyckDataset(n_examples=args.train_examples, seed=args.dataset_seed, **common),
            DyckDataset(n_examples=args.val_examples, seed=args.dataset_seed + 100_000, **common),
        )
    if args.task == "unification":
        return (
            UnificationDataset(n_examples=args.train_examples, seed=args.dataset_seed, depth=args.depth),
            UnificationDataset(n_examples=args.val_examples, seed=args.dataset_seed + 100_000, depth=args.depth),
        )
    if args.task == "graph_alias":
        return build_graph_alias(args)
    if args.task == "causal_intervention":
        common = {"n_events": args.n_events, "edge_prob": args.edge_prob}
        return (
            CausalInterventionDataset(n_examples=args.train_examples, seed=args.dataset_seed, **common),
            CausalInterventionDataset(n_examples=args.val_examples, seed=args.dataset_seed + 100_000, **common),
        )
    if args.task == "structural_paraphrase":
        return (
            StructuralParaphraseDataset(n_examples=args.train_examples, seed=args.dataset_seed),
            StructuralParaphraseDataset(n_examples=args.val_examples, seed=args.dataset_seed + 100_000),
        )
    if args.task == "temporal_query":
        common = {"n_events": args.n_events, "edge_prob": args.edge_prob}
        return (
            TemporalQueryDataset(n_examples=args.train_examples, seed=args.dataset_seed, **common),
            TemporalQueryDataset(n_examples=args.val_examples, seed=args.dataset_seed + 100_000, **common),
        )
    raise ValueError(f"unknown task: {args.task}")


def main() -> None:
    args = parse_args()
    enable_deterministic(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(args)
    tokenizer = StoryTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(train_ds.all_texts() + val_ds.all_texts() + ["yes no"])

    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    runs = []
    for seed in args.seeds:
        for condition in conditions:
            runs.append(train_condition(args, condition, seed, train_ds, val_ds, tokenizer))

    balance = label_balance(val_ds)
    summary = summarise(runs, balance["majority_acc"])
    results = {
        "task": args.task,
        "summary": summary,
        "verdict": regime_verdict(summary),
        "label_balance": balance,
        "runs": runs,
        "config": vars(args),
    }
    (args.output_dir / "results.json").write_text(json.dumps(jsonify(results), indent=2))
    write_report(args.output_dir / "report.md", args, runs, balance)
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
