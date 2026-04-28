#!/usr/bin/env python3
"""Regime probe for aliased graph-walk questions.

The task is deliberately designed to defeat lexical overlap shortcuts.  Each
example describes a latent directed graph, alias equivalence classes for its
nodes, a walk prefix rendered with non-canonical aliases, and one candidate
next node rendered with another alias.  The answer is whether the candidate is
an outgoing neighbor of the current latent node.

This script reuses ``regime_probe.py`` diagnostics so the acceptance gate is
the same across temporal stories and graph-alias tasks.
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
from synthetic.graph_alias import GraphAliasDataset  # noqa: E402
from synthetic.semantic_story import StoryTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/graph_alias_probe"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", default="standard,resonance_full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train_examples", type=int, default=512)
    parser.add_argument("--val_examples", type=int, default=256)
    parser.add_argument("--n_graphs", type=int, default=16)
    parser.add_argument("--n_nodes", type=int, default=16)
    parser.add_argument("--aliases_per_node", type=int, default=3)
    parser.add_argument("--alias_pool_size", type=int, default=0)
    parser.add_argument("--out_degree", type=int, default=2)
    parser.add_argument("--walk_length", type=int, default=4)
    parser.add_argument("--query_steps", type=int, default=1)
    parser.add_argument(
        "--val_graphs",
        choices=["same", "same_aliases_new_edges", "new"],
        default="new",
        help="Use held-out walks over training graphs, rewired graphs with known aliases, or entirely new graphs/aliases.",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--phase_init_std", type=float, default=0.3)
    parser.add_argument("--story_phase_prior_std", type=float, default=1.2)
    parser.add_argument("--story_phase_prior_noise", type=float, default=0.05)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--dataset_seed", type=int, default=7000)
    parser.add_argument("--dispersion_lambda", type=float, default=0.0)
    parser.add_argument("--eval_each_epoch", action="store_true")
    return parser.parse_args()


def build_datasets(args: argparse.Namespace) -> tuple[GraphAliasDataset, GraphAliasDataset]:
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
    train_ds = GraphAliasDataset(
        n_examples=args.train_examples,
        seed=args.dataset_seed,
        **common,
    )
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
