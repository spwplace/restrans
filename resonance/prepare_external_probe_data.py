#!/usr/bin/env python3
"""Convert staged external datasets into generic yes/no structural probes.

The outputs are intentionally simple JSONL files:

    {"prefix": "... Answer", "answer": "yes"|"no", "id": "...", "source": "..."}

They are consumed by:

    resonance/structural_task_probe.py --task generic_probe

These converters are not meant to be final benchmark definitions.  They make
the second wave executable so we can discover which external datasets are worth
turning into stricter task-specific evaluations.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_from_disk


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], limit: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def clean_text(value: Any, max_chars: int | None = None) -> str:
    text = " ".join(str(value).replace("\r", "\n").split())
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + " ..."
    return text


def yes_no(value: bool | int | str) -> str:
    if isinstance(value, str):
        return "yes" if value.lower() in {"yes", "true", "1", "entailment"} else "no"
    return "yes" if bool(value) else "no"


def alternating_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    input_field: str,
    target_field: str,
    prompt_name: str,
    id_field: str | None = None,
    max_input_chars: int | None = None,
    max_target_chars: int | None = None,
    seed: int = 0,
) -> Iterable[dict[str, Any]]:
    rng = random.Random(seed)
    targets = [row[target_field] for row in rows]
    n = len(rows)
    for idx, row in enumerate(rows):
        positive = idx % 2 == 0
        target = row[target_field]
        if not positive and n > 1:
            neg_idx = rng.randrange(n - 1)
            if neg_idx >= idx:
                neg_idx += 1
            target = targets[neg_idx]
        prefix = (
            f"{prompt_name} Input: {clean_text(row[input_field], max_input_chars)} "
            f"Candidate meaning: {clean_text(target, max_target_chars)} "
            "Is the candidate meaning correct? Answer"
        )
        yield {
            "prefix": prefix,
            "answer": "yes" if positive else "no",
            "id": str(row.get(id_field, idx)) if id_field else str(idx),
            "source": source,
            "metadata": {"positive": positive},
        }


def dataset_split_rows(ds: Dataset | DatasetDict, split: str) -> list[dict[str, Any]]:
    part = ds[split] if isinstance(ds, DatasetDict) else ds
    return [dict(part[idx]) for idx in range(len(part))]


def convert_hans(args: argparse.Namespace) -> dict[str, int]:
    root = args.data_dir / "source_repos/hans"
    out = args.output_dir / "hans"
    counts = {}
    for split, filename in [
        ("train", "heuristics_train_set.jsonl"),
        ("evaluation", "heuristics_evaluation_set.jsonl"),
    ]:
        source_path = root / filename
        if not source_path.exists():
            continue

        def rows() -> Iterable[dict[str, Any]]:
            with source_path.open() as handle:
                for idx, line in enumerate(handle):
                    row = json.loads(line)
                    yield {
                        "prefix": (
                            f"Premise: {clean_text(row['sentence1'])} "
                            f"Hypothesis: {clean_text(row['sentence2'])} "
                            "Does the premise entail the hypothesis? Answer"
                        ),
                        "answer": yes_no(row["gold_label"]),
                        "id": str(row.get("pairID", idx)),
                        "source": "hans",
                        "metadata": {
                            "heuristic": row.get("heuristic"),
                            "subcase": row.get("subcase"),
                            "template": row.get("template"),
                        },
                    }

        counts[f"hans/{split}"] = write_jsonl(out / f"{split}.jsonl", rows(), args.limit)
    return counts


def convert_msgs(args: argparse.Namespace) -> dict[str, int]:
    root = args.data_dir / "processed/msgs/msgs"
    out = args.output_dir / "msgs"
    counts = {}
    if not root.exists():
        return counts
    for source_path in sorted(root.glob("*/*.jsonl")):
        paradigm = source_path.parent.name
        split = source_path.stem

        def rows(path: Path = source_path) -> Iterable[dict[str, Any]]:
            with path.open() as handle:
                for idx, line in enumerate(handle):
                    row = json.loads(line)
                    question = row.get("linguistic_feature_description") or "Does the sentence have the target linguistic feature?"
                    yield {
                        "prefix": f"Sentence: {clean_text(row['sentence'])} Question: {question} Answer",
                        "answer": yes_no(row["linguistic_feature_label"]),
                        "id": str(row.get("sentenceID", idx)),
                        "source": "msgs",
                        "metadata": {
                            "paradigm": paradigm,
                            "split": split,
                            "uid": row.get("UID"),
                            "surface_feature_label": row.get("surface_feature_label"),
                            "control_paradigm": row.get("control_paradigm"),
                        },
                    }

        counts[f"msgs/{paradigm}/{split}"] = write_jsonl(out / paradigm / f"{split}.jsonl", rows(), args.limit)
    return counts


def convert_slog(args: argparse.Namespace) -> dict[str, int]:
    counts = {}
    sources = [
        ("slog_cogs_lf_train", args.data_dir / "source_repos/slog/data/cogs_LF/train.tsv"),
        ("slog_cogs_lf_dev", args.data_dir / "source_repos/slog/data/cogs_LF/dev.tsv"),
        ("slog_cogs_lf_test", args.data_dir / "source_repos/slog/data/cogs_LF/test.tsv"),
        (
            "slog_gen_cogs_lf",
            args.data_dir / "processed/slog_generalization_sets/generalization_sets/gen_cogsLF.tsv",
        ),
        (
            "slog_gen_varfree_lf",
            args.data_dir / "processed/slog_generalization_sets/generalization_sets/gen_varfreeLF.tsv",
        ),
    ]
    out = args.output_dir / "slog"
    for name, source_path in sources:
        if not source_path.exists():
            continue
        raw_rows = []
        with source_path.open() as handle:
            for idx, line in enumerate(handle):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    raw_rows.append(
                        {
                            "id": idx,
                            "sentence": parts[0],
                            "logical_form": parts[1],
                            "domain": parts[2] if len(parts) > 2 else None,
                        }
                    )
        counts[f"slog/{name}"] = write_jsonl(
            out / f"{name}.jsonl",
            alternating_candidate_rows(
                raw_rows,
                source=name,
                input_field="sentence",
                target_field="logical_form",
                prompt_name="Semantic parse probe.",
                id_field="id",
                max_input_chars=args.max_text_chars,
                max_target_chars=args.max_target_chars,
                seed=args.seed,
            ),
            args.limit,
        )
    return counts


def convert_cogs(args: argparse.Namespace) -> dict[str, int]:
    path = args.data_dir / "hf_datasets/cogs"
    if not path.exists():
        return {}
    ds = load_from_disk(str(path))
    out = args.output_dir / "cogs"
    counts = {}
    for split in ds.keys():
        rows = [{"id": idx, **dict(ds[split][idx])} for idx in range(len(ds[split]))]
        counts[f"cogs/{split}"] = write_jsonl(
            out / f"{split}.jsonl",
            alternating_candidate_rows(
                rows,
                source=f"cogs/{split}",
                input_field="input",
                target_field="output",
                prompt_name="Semantic parse probe.",
                id_field="id",
                max_input_chars=args.max_text_chars,
                max_target_chars=args.max_target_chars,
                seed=args.seed,
            ),
            args.limit,
        )
    return counts


def convert_cfq(args: argparse.Namespace) -> dict[str, int]:
    counts = {}
    out = args.output_dir / "cfq"
    for config_dir in sorted((args.data_dir / "hf_datasets/cfq").glob("*")):
        if not config_dir.is_dir():
            continue
        ds = load_from_disk(str(config_dir))
        for split in ds.keys():
            rows = [{"id": idx, **dict(ds[split][idx])} for idx in range(len(ds[split]))]
            counts[f"cfq/{config_dir.name}/{split}"] = write_jsonl(
                out / config_dir.name / f"{split}.jsonl",
                alternating_candidate_rows(
                    rows,
                    source=f"cfq/{config_dir.name}/{split}",
                    input_field="question",
                    target_field="query",
                    prompt_name="Semantic query probe.",
                    id_field="id",
                    max_input_chars=args.max_text_chars,
                    max_target_chars=args.max_target_chars,
                    seed=args.seed,
                ),
                args.limit,
            )
    return counts


def convert_poj104(args: argparse.Namespace) -> dict[str, int]:
    path = args.data_dir / "hf_datasets/optional/google__code_x_glue_cc_clone_detection_poj104"
    if not path.exists():
        return {}
    ds = load_from_disk(str(path))
    labels = sorted(
        {
            str(ds[split][idx]["label"])
            for split in ds.keys()
            for idx in range(len(ds[split]))
        }
    )
    out = args.output_dir / "poj104"
    counts = {}
    for split in ds.keys():
        part = ds[split]

        def rows() -> Iterable[dict[str, Any]]:
            for idx in range(len(part)):
                row = part[idx]
                correct = str(row["label"])
                positive = idx % 2 == 0
                candidate = correct
                if not positive:
                    candidate = labels[(labels.index(correct) + 17) % len(labels)]
                    if candidate == correct:
                        candidate = labels[(labels.index(correct) + 1) % len(labels)]
                yield {
                    "prefix": (
                        f"Program: {clean_text(row['code'], args.max_code_chars)} "
                        f"Candidate problem id: {candidate}. Does the program solve this problem? Answer"
                    ),
                    "answer": "yes" if positive else "no",
                    "id": str(row.get("id", idx)),
                    "source": f"poj104/{split}",
                    "metadata": {"true_label": correct, "candidate": candidate},
                }

        counts[f"poj104/{split}"] = write_jsonl(out / f"{split}.jsonl", rows(), args.limit)
    return counts


def convert_bigclonebench(args: argparse.Namespace) -> dict[str, int]:
    path = args.data_dir / "hf_datasets/optional/google__code_x_glue_cc_clone_detection_big_clone_bench"
    if not path.exists():
        return {}
    ds = load_from_disk(str(path))
    out = args.output_dir / "bigclonebench"
    counts = {}
    for split in ds.keys():
        part = ds[split]

        def rows() -> Iterable[dict[str, Any]]:
            for idx in range(len(part)):
                row = part[idx]
                yield {
                    "prefix": (
                        f"Function one: {clean_text(row['func1'], args.max_code_chars)} "
                        f"Function two: {clean_text(row['func2'], args.max_code_chars)} "
                        "Are these functions semantic clones? Answer"
                    ),
                    "answer": yes_no(row["label"]),
                    "id": str(row.get("id", idx)),
                    "source": f"bigclonebench/{split}",
                }

        counts[f"bigclonebench/{split}"] = write_jsonl(out / f"{split}.jsonl", rows(), args.limit)
    return counts


def convert_codesearchnet(args: argparse.Namespace) -> dict[str, int]:
    path = args.data_dir / "hf_datasets/optional/sentence-transformers__codesearchnet__pair"
    if not path.exists():
        return {}
    ds = load_from_disk(str(path))
    part = ds["train"]
    n = min(len(part), args.limit or len(part))
    sampled = [dict(part[idx]) for idx in range(n)]
    out = args.output_dir / "codesearchnet"
    count = write_jsonl(
        out / "train.jsonl",
        alternating_candidate_rows(
            [{"id": idx, **row} for idx, row in enumerate(sampled)],
            source="codesearchnet/train",
            input_field="comment",
            target_field="code",
            prompt_name="Code-search probe.",
            id_field="id",
            max_input_chars=args.max_text_chars,
            max_target_chars=args.max_code_chars,
            seed=args.seed,
        ),
        args.limit,
    )
    return {"codesearchnet/train": count}


CONVERTERS = {
    "hans": convert_hans,
    "msgs": convert_msgs,
    "slog": convert_slog,
    "cogs": convert_cogs,
    "cfq": convert_cfq,
    "poj104": convert_poj104,
    "bigclonebench": convert_bigclonebench,
    "codesearchnet": convert_codesearchnet,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/probes"))
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", *CONVERTERS.keys()])
    parser.add_argument("--limit", type=int, default=10_000, help="Max rows per output split/paradigm.")
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--max_text_chars", type=int, default=600)
    parser.add_argument("--max_target_chars", type=int, default=900)
    parser.add_argument("--max_code_chars", type=int, default=1600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(CONVERTERS) if "all" in args.datasets else args.datasets
    manifest: dict[str, int] = {}
    for name in selected:
        print(f"=== {name} ===", flush=True)
        counts = CONVERTERS[name](args)
        for key, value in sorted(counts.items()):
            print(f"{key}: {value}", flush=True)
        manifest.update(counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
