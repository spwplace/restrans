#!/usr/bin/env python3
"""Download and stage public datasets for the resonance research programme.

The script writes Hugging Face datasets to ``data/hf_datasets`` and small
JSONL exports to ``data/processed`` where useful.  It records every success or
failure in ``data/dataset_manifest.json`` so interrupted runs are easy to
inspect.

AMR note: the canonical AMR 3.0 release is LDC2020T02 and requires the LDC
license.  This script can fetch a public Hugging Face derivative if available,
but it records that separately from canonical licensed AMR.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import tarfile
import traceback
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, DatasetDict, concatenate_datasets, get_dataset_config_names, load_dataset
from huggingface_hub import snapshot_download


@dataclass
class DatasetStatus:
    name: str
    status: str
    path: str | None = None
    rows: dict[str, int] | None = None
    note: str | None = None
    error: str | None = None


def rows_of(ds: Dataset | DatasetDict) -> dict[str, int]:
    if isinstance(ds, DatasetDict):
        return {split: len(part) for split, part in ds.items()}
    return {"train": len(ds)}


def save_dataset(ds: Dataset | DatasetDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    ds.save_to_disk(str(path))


def export_jsonl(ds: Dataset, path: Path, limit: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        n = len(ds) if limit is None else min(limit, len(ds))
        for idx in range(n):
            handle.write(json.dumps(ds[idx], ensure_ascii=False) + "\n")


def upsert_status(statuses: list[DatasetStatus], status: DatasetStatus) -> None:
    statuses[:] = [old for old in statuses if old.name != status.name]
    statuses.append(status)


def load_manifest_statuses(path: Path) -> list[DatasetStatus]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    statuses = []
    superseded_failures = {
        "optional:scan",
        "optional:scan/simple",
        "optional:scan/length",
        "optional:google/cfq",
        "optional:cfq",
    }
    for item in raw.get("statuses", []):
        if item.get("status") == "failed" and item.get("name") in superseded_failures:
            continue
        allowed = {field.name for field in DatasetStatus.__dataclass_fields__.values()}
        statuses.append(DatasetStatus(**{key: value for key, value in item.items() if key in allowed}))
    return statuses


def run_step(name: str, statuses: list[DatasetStatus], fn: Callable[[], DatasetStatus]) -> None:
    print(f"\n=== {name} ===", flush=True)
    try:
        status = fn()
        print(f"{name}: {status.status}", flush=True)
        upsert_status(statuses, status)
    except Exception as exc:  # noqa: BLE001 - downloader should continue.
        traceback.print_exc()
        upsert_status(statuses, DatasetStatus(name=name, status="failed", error=f"{type(exc).__name__}: {exc}"))


def download_wikitext(args: argparse.Namespace, config: str) -> DatasetStatus:
    ds = load_dataset("Salesforce/wikitext", config, cache_dir=str(args.cache_dir))
    out = args.hf_dir / "wikitext" / config
    save_dataset(ds, out)
    return DatasetStatus(
        name=f"wikitext/{config}",
        status="ok",
        path=str(out),
        rows=rows_of(ds),
        note="Public WikiText language-modeling corpus.",
    )


def download_tinystories(args: argparse.Namespace) -> DatasetStatus:
    if args.tinystories_full:
        ds = load_dataset("roneneldan/TinyStories", cache_dir=str(args.cache_dir))
    else:
        train = load_dataset(
            "roneneldan/TinyStories",
            split=f"train[:{args.tinystories_train}]",
            cache_dir=str(args.cache_dir),
        )
        validation = load_dataset(
            "roneneldan/TinyStories",
            split=f"validation[:{args.tinystories_val}]",
            cache_dir=str(args.cache_dir),
        )
        ds = DatasetDict({"train": train, "validation": validation})
    out = args.hf_dir / "tinystories"
    save_dataset(ds, out)
    return DatasetStatus(
        name="roneneldan/TinyStories",
        status="ok",
        path=str(out),
        rows=rows_of(ds),
        note="Subset by default; use --tinystories_full for full multi-GB corpus.",
    )


def download_blimp(args: argparse.Namespace) -> DatasetStatus:
    configs = get_dataset_config_names("nyu-mll/blimp", cache_dir=str(args.cache_dir))
    pieces = []
    root = args.hf_dir / "blimp"
    processed = args.processed_dir / "blimp"
    for config in configs:
        ds = load_dataset("nyu-mll/blimp", config, split="train", cache_dir=str(args.cache_dir))
        save_dataset(ds, root / config)
        export_jsonl(ds, processed / f"{config}.jsonl")
        pieces.append(ds)
    combined = concatenate_datasets(pieces)
    save_dataset(combined, root / "all")
    export_jsonl(combined, processed / "all.jsonl")
    return DatasetStatus(
        name="nyu-mll/blimp",
        status="ok",
        path=str(root),
        rows={"all": len(combined), "configs": len(configs)},
        note="67 BLiMP paradigms exported as per-paradigm and combined JSONL.",
    )


def download_cogs(args: argparse.Namespace) -> DatasetStatus:
    ds = load_dataset("Punchwe/COGS", cache_dir=str(args.cache_dir))
    out = args.hf_dir / "cogs"
    save_dataset(ds, out)
    processed = args.processed_dir / "cogs"
    if isinstance(ds, DatasetDict):
        for split, part in ds.items():
            export_jsonl(part, processed / f"{split}.jsonl")
    return DatasetStatus(
        name="Punchwe/COGS",
        status="ok",
        path=str(out),
        rows=rows_of(ds),
        note="Compositional generalization semantic parsing benchmark.",
    )


def download_amr_public_derivative(args: argparse.Namespace) -> DatasetStatus:
    ds = load_dataset("hoshuhan/amr-3-parsed", cache_dir=str(args.cache_dir))
    out = args.hf_dir / "amr_3_parsed_public_derivative"
    save_dataset(ds, out)
    processed = args.processed_dir / "amr_3_parsed_public_derivative"
    processed.mkdir(parents=True, exist_ok=True)
    if isinstance(ds, DatasetDict):
        for split, part in ds.items():
            export_jsonl(part, processed / f"{split}.jsonl")
    return DatasetStatus(
        name="hoshuhan/amr-3-parsed",
        status="ok",
        path=str(out),
        rows=rows_of(ds),
        note=(
            "Public HF derivative formatted as conversations. Canonical AMR 3.0 is "
            "LDC2020T02 and should be used under LDC terms for serious AMR claims."
        ),
    )


def export_babylm_split(root: Path, split_dir: str, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w") as handle:
        for source_path in sorted((root / split_dir).glob("*.txt")):
            with source_path.open(encoding="utf-8") as source:
                filename = source.readline().strip()
                while True:
                    text = source.readline()
                    tagged = source.readline()
                    if not text or not tagged:
                        break
                    row = {
                        "text": text.strip(),
                        "tagged_text": tagged.strip(),
                        "filename": filename or source_path.name,
                        "source_path": str(source_path.relative_to(root)),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows += 1
    return rows


def download_babylm_strict_small(args: argparse.Namespace) -> DatasetStatus:
    root = args.hf_dir / "babylm_strict_small_snapshot"
    patterns = [
        "README.md",
        "BabyLM.py",
        "clean_tagged/10M/*.txt",
        "clean_tagged/dev/*.txt",
        "clean_tagged/test/*.txt",
    ]
    snapshot_download(
        repo_id="cambridge-climb/BabyLM",
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(root),
        cache_dir=str(args.cache_dir),
    )
    processed = args.processed_dir / "babylm_strict_small"
    rows = {
        "train": export_babylm_split(root, "clean_tagged/10M", processed / "train.jsonl"),
        "validation": export_babylm_split(root, "clean_tagged/dev", processed / "validation.jsonl"),
        "test": export_babylm_split(root, "clean_tagged/test", processed / "test.jsonl"),
    }
    return DatasetStatus(
        name="cambridge-climb/BabyLM/strict_small_snapshot",
        status="ok",
        path=str(root),
        rows=rows,
        note=(
            "Fetched only the cleaned 10M-word strict-small BabyLM files plus dev/test, "
            "then exported line-pair JSONL. This avoids downloading the full multi-config HF repo."
        ),
    )


def download_babylm_full_snapshot(args: argparse.Namespace) -> DatasetStatus:
    root = args.hf_dir / "babylm_full_snapshot"
    snapshot_download(
        repo_id="cambridge-climb/BabyLM",
        repo_type="dataset",
        local_dir=str(root),
        cache_dir=str(args.cache_dir),
    )
    return DatasetStatus(
        name="cambridge-climb/BabyLM/full_snapshot",
        status="ok",
        path=str(root),
        rows=summarize_source_tree(root),
        note=(
            "Full BabyLM dataset repository snapshot. This is intentionally a raw snapshot; "
            "strict-small JSONL exports live under processed/babylm_strict_small."
        ),
    )


def summarize_source_tree(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {"files": len(files)}


def download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, target)  # noqa: S310 - URLs are fixed dataset sources.


def extract_tar_gz(archive: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        return
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(out)  # noqa: S202 - fixed dataset archive, local research workflow.


def extract_zip(archive: Path, out: Path, pwd: bytes | None = None) -> None:
    if out.exists() and any(item.is_file() for item in out.rglob("*")):
        return
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out, pwd=pwd)


def download_linzen_simple(args: argparse.Namespace) -> DatasetStatus:
    archive = args.archive_dir / "rnn_agreement" / "rnn_agr_simple.tar.gz"
    out = args.processed_dir / "rnn_agreement_simple"
    download_url("http://tallinzen.net/media/rnn_agreement/rnn_agr_simple.tar.gz", archive)
    extract_tar_gz(archive, out)
    return DatasetStatus(
        name="linzen/rnn_agr_simple",
        status="ok",
        path=str(out),
        rows=summarize_source_tree(out),
        note="Simple dependency dataset linked from TalLinzen/rnn_agreement README.",
    )


def download_linzen_dependency_tsv(args: argparse.Namespace) -> DatasetStatus:
    archive = args.archive_dir / "rnn_agreement" / "agr_50_mostcommon_10K.tsv.gz"
    out = args.processed_dir / "rnn_agreement" / "agr_50_mostcommon_10K.tsv"
    download_url("https://www.dropbox.com/s/5axdu3q9jkxzjy8/agr_50_mostcommon_10K.tsv.gz?dl=1", archive)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        with gzip.open(archive, "rb") as source, out.open("wb") as target:
            shutil.copyfileobj(source, target)
    rows = 0
    with out.open(errors="replace") as handle:
        for rows, _line in enumerate(handle, start=1):
            pass
    return DatasetStatus(
        name="linzen/agr_50_mostcommon_10K",
        status="ok",
        path=str(out),
        rows={"lines": rows},
        note="Dependency TSV/GZ linked from TalLinzen/rnn_agreement README.",
    )


def extract_msgs_zip(args: argparse.Namespace) -> DatasetStatus:
    archive = args.source_dir / "msgs" / "data" / "msgs.zip"
    out = args.processed_dir / "msgs"
    extract_zip(archive, out)
    return DatasetStatus(
        name="msgs/extracted",
        status="ok",
        path=str(out),
        rows=summarize_source_tree(out),
        note="Extracted MSGS zip from nyu-mll/msgs.",
    )


def extract_slog_generalization(args: argparse.Namespace) -> DatasetStatus:
    archive = args.source_dir / "slog" / "data" / "generalization_sets.zip"
    out = args.processed_dir / "slog_generalization_sets"
    extract_zip(archive, out, pwd=b"SLOG")
    return DatasetStatus(
        name="slog/generalization_sets",
        status="ok",
        path=str(out),
        rows=summarize_source_tree(out),
        note="Extracted SLOG generalization sets from bingzhilee/slog.",
    )


def stage_git_repo(args: argparse.Namespace, name: str, url: str, note: str | None = None) -> DatasetStatus:
    out = args.source_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(out)],
            check=True,
            text=True,
            capture_output=True,
        )
    return DatasetStatus(
        name=f"github:{name}",
        status="ok",
        path=str(out),
        rows=summarize_source_tree(out),
        note=note or f"Staged source repository from {url}.",
    )


def download_cfq_config(args: argparse.Namespace, config: str) -> DatasetStatus:
    ds = load_dataset("cfq", config, cache_dir=str(args.cache_dir))
    out = args.hf_dir / "cfq" / config
    save_dataset(ds, out)
    processed = args.processed_dir / "cfq" / config
    if isinstance(ds, DatasetDict):
        for split, part in ds.items():
            export_jsonl(part, processed / f"{split}.jsonl", limit=args.optional_export_limit)
    return DatasetStatus(
        name=f"cfq/{config}",
        status="ok",
        path=str(out),
        rows=rows_of(ds),
        note="Compositional Freebase Questions split; JSONL export is capped by --optional_export_limit.",
    )


def download_hf_named(
    args: argparse.Namespace,
    repo: str,
    *,
    config: str | None = None,
    local_name: str | None = None,
    note: str | None = None,
    export_limit: int | None = None,
) -> DatasetStatus:
    ds = load_dataset(repo, config, cache_dir=str(args.cache_dir)) if config else load_dataset(repo, cache_dir=str(args.cache_dir))
    label = f"{repo}/{config}" if config else repo
    safe = (local_name or label).replace("/", "__")
    out = args.hf_dir / "optional" / safe
    save_dataset(ds, out)
    processed = args.processed_dir / "optional" / safe
    if isinstance(ds, DatasetDict):
        for split, part in ds.items():
            export_jsonl(part, processed / f"{split}.jsonl", limit=export_limit)
    elif export_limit is not None:
        export_jsonl(ds, processed / "data.jsonl", limit=export_limit)
    return DatasetStatus(
        name=label,
        status="ok",
        path=str(out),
        rows=rows_of(ds),
        note=note,
    )


def download_equibench(args: argparse.Namespace) -> DatasetStatus:
    configs = ["DCE", "OJ_A", "OJ_V", "OJ_VA", "STOKE", "TVM"]
    root = args.hf_dir / "equibench"
    processed = args.processed_dir / "equibench"
    rows: dict[str, int] = {}
    for config in configs:
        ds = load_dataset("anjiangwei/EquiBench-Datasets", config, cache_dir=str(args.cache_dir))
        out = root / config
        save_dataset(ds, out)
        if isinstance(ds, DatasetDict):
            for split, part in ds.items():
                rows[f"{config}/{split}"] = len(part)
                export_jsonl(part, processed / config / f"{split}.jsonl", limit=args.optional_export_limit)
        else:
            rows[config] = len(ds)
            export_jsonl(ds, processed / config / "data.jsonl", limit=args.optional_export_limit)
    return DatasetStatus(
        name="anjiangwei/EquiBench-Datasets",
        status="ok",
        path=str(root),
        rows=rows,
        note=(
            "Public equivalence-checking benchmark with six configs. JSONL export "
            "is capped by --optional_export_limit; saved HF datasets are complete."
        ),
    )


def try_optional_hf(args: argparse.Namespace, repo: str, config: str | None = None) -> DatasetStatus:
    label = f"{repo}/{config}" if config else repo
    ds = load_dataset(repo, config, cache_dir=str(args.cache_dir)) if config else load_dataset(repo, cache_dir=str(args.cache_dir))
    safe = label.replace("/", "__")
    out = args.hf_dir / "optional" / safe
    save_dataset(ds, out)
    return DatasetStatus(name=label, status="ok", path=str(out), rows=rows_of(ds))


def write_linzen_style_generated(args: argparse.Namespace) -> DatasetStatus:
    from synthetic.agreement import AgreementDataset

    root = args.processed_dir / "linzen_style_generated"
    root.mkdir(parents=True, exist_ok=True)
    train = AgreementDataset(
        n_examples=args.linzen_style_train,
        seed=args.dataset_seed,
        max_attractors=args.linzen_max_attractors,
    )
    validation = AgreementDataset(
        n_examples=args.linzen_style_val,
        seed=args.dataset_seed + 100_000,
        max_attractors=args.linzen_max_attractors,
    )
    for name, ds in [("train", train), ("validation", validation)]:
        with (root / f"{name}.jsonl").open("w") as handle:
            for example in ds.examples:
                handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    note = (
        "Generated Linzen-style attractor agreement data. This is not the original "
        "Linzen et al. corpus; use BLiMP or a manually supplied original file for "
        "literature-faithful claims."
    )
    (root / "README.md").write_text(note + "\n")
    return DatasetStatus(
        name="linzen_style_generated",
        status="ok",
        path=str(root),
        rows={"train": len(train), "validation": len(validation)},
        note=note,
    )


def write_manual_notes(args: argparse.Namespace) -> None:
    manual = args.data_dir / "manual"
    manual.mkdir(parents=True, exist_ok=True)
    (manual / "AMR_LDC2020T02.md").write_text(
        "# AMR 3.0 Canonical Data\n\n"
        "Canonical AMR 3.0 is LDC2020T02 and requires LDC access. Put the licensed "
        "release here if available, then add a local converter. The downloader also "
        "tries `hoshuhan/amr-3-parsed`, a public Hugging Face derivative, but do not "
        "treat that as a substitute for licensed AMR in formal claims without checking rights.\n"
    )
    (manual / "linzen_original.md").write_text(
        "# Linzen et al. Agreement Data\n\n"
        "I did not rely on an unstable direct download for the original Linzen et al. "
        "number-prediction corpus. Use `agreement_file` with a local JSONL/CSV if you "
        "obtain the original data. The downloader creates `processed/linzen_style_generated` "
        "as a compatible controlled substitute.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--profile", choices=["core", "full"], default="core")
    parser.add_argument("--dataset_seed", type=int, default=9000)
    parser.add_argument("--tinystories_train", type=int, default=100_000, help="Subset rows staged when --tinystories_full is not set.")
    parser.add_argument("--tinystories_val", type=int, default=5_000, help="Validation subset rows staged when --tinystories_full is not set.")
    parser.add_argument("--tinystories_full", action="store_true", help="Download/stage the full TinyStories dataset instead of the default fast subset.")
    parser.add_argument("--linzen_style_train", type=int, default=100_000, help="Rows generated for the synthetic Linzen-style training split.")
    parser.add_argument("--linzen_style_val", type=int, default=10_000, help="Rows generated for the synthetic Linzen-style validation split.")
    parser.add_argument("--linzen_max_attractors", type=int, default=5)
    parser.add_argument(
        "--optional_export_limit",
        type=int,
        default=10_000,
        help=(
            "Cap only the convenience JSONL exports for very large optional datasets. "
            "The saved Hugging Face dataset is not truncated."
        ),
    )
    parser.add_argument("--cfq_configs", default="mcd1,mcd2,mcd3")
    parser.add_argument("--skip_core", action="store_true")
    parser.add_argument("--skip_optional", action="store_true")
    parser.add_argument("--skip_github", action="store_true")
    parser.add_argument("--skip_babylm", action="store_true")
    parser.add_argument("--babylm_full", action="store_true", help="Also snapshot the full BabyLM dataset repository.")
    parser.add_argument("--skip_code", action="store_true")
    parser.add_argument("--include_codesearchnet", action="store_true")
    parser.add_argument("--include_big_code", action="store_true")
    parser.add_argument("--skip_equibench", action="store_true")
    parser.add_argument("--only_equibench", action="store_true", help="Only stage EquiBench and update the manifest.")
    parser.add_argument("--include_leandojo", action="store_true", help="Clone LeanDojo-v2 source for proof-progress setup. Installation is handled by scripts/setup_leandojo_progress.sh.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = args.data_dir / "hf_cache"
    args.hf_dir = args.data_dir / "hf_datasets"
    args.processed_dir = args.data_dir / "processed"
    args.source_dir = args.data_dir / "source_repos"
    args.archive_dir = args.data_dir / "source_archives"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.hf_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.data_dir / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.cache_dir))

    manifest_path = args.data_dir / "dataset_manifest.json"
    statuses: list[DatasetStatus] = load_manifest_statuses(manifest_path)
    write_manual_notes(args)

    if args.only_equibench:
        run_step("equibench", statuses, lambda: download_equibench(args))
        manifest = {
            "data_dir": str(args.data_dir),
            "profile": args.profile,
            "statuses": [asdict(status) for status in statuses],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\nWrote {manifest_path}")
        return

    if not args.skip_core:
        run_step("wikitext-2-raw-v1", statuses, lambda: download_wikitext(args, "wikitext-2-raw-v1"))
        run_step("wikitext-103-raw-v1", statuses, lambda: download_wikitext(args, "wikitext-103-raw-v1"))
        run_step("tinystories", statuses, lambda: download_tinystories(args))
        run_step("blimp", statuses, lambda: download_blimp(args))
        run_step("cogs", statuses, lambda: download_cogs(args))
        run_step("amr_public_derivative", statuses, lambda: download_amr_public_derivative(args))
        run_step("linzen_style_generated", statuses, lambda: write_linzen_style_generated(args))

    if not args.skip_babylm:
        run_step("babylm_strict_small", statuses, lambda: download_babylm_strict_small(args))
        if args.babylm_full:
            run_step("babylm_full", statuses, lambda: download_babylm_full_snapshot(args))

    if not args.skip_github:
        github_sources: list[tuple[str, str, str]] = [
            (
                "rnn_agreement",
                "https://github.com/TalLinzen/rnn_agreement.git",
                "Linzen et al. subject-verb agreement code/data repository.",
            ),
            (
                "slog",
                "https://github.com/bingzhilee/slog.git",
                "SLOG structural long-distance generalization benchmark source repository.",
            ),
            (
                "hans",
                "https://github.com/tommccoy1/hans.git",
                "HANS syntactic-heuristic NLI evaluation/train sets.",
            ),
            (
                "msgs",
                "https://github.com/nyu-mll/msgs.git",
                "MSGs minimal generalization split repository.",
            ),
            (
                "scan",
                "https://github.com/brendenlake/SCAN.git",
                "SCAN compositional command-following benchmark source repository.",
            ),
        ]
        for name, url, note in github_sources:
            run_step(f"github:{name}", statuses, lambda name=name, url=url, note=note: stage_git_repo(args, name, url, note))
        run_step("linzen/rnn_agr_simple", statuses, lambda: download_linzen_simple(args))
        run_step("linzen/agr_50_mostcommon_10K", statuses, lambda: download_linzen_dependency_tsv(args))
        run_step("msgs/extracted", statuses, lambda: extract_msgs_zip(args))
        run_step("slog/generalization_sets", statuses, lambda: extract_slog_generalization(args))

    if not args.skip_optional:
        for config in [item.strip() for item in args.cfq_configs.split(",") if item.strip()]:
            run_step(f"cfq/{config}", statuses, lambda config=config: download_cfq_config(args, config))

    if not args.skip_code:
        if not args.skip_equibench:
            run_step("equibench", statuses, lambda: download_equibench(args))
        run_step(
            "code_x_glue_poj104",
            statuses,
            lambda: download_hf_named(
                args,
                "google/code_x_glue_cc_clone_detection_poj104",
                note="C-UDA CodeXGLUE POJ-104 problem-classification dataset.",
                export_limit=args.optional_export_limit,
            ),
        )
        if args.include_codesearchnet:
            run_step(
                "codesearchnet_pair",
                statuses,
                lambda: download_hf_named(
                    args,
                    "sentence-transformers/codesearchnet",
                    config="pair",
                    note="CodeSearchNet comment-code pairs; JSONL export is capped by --optional_export_limit.",
                    export_limit=args.optional_export_limit,
                ),
            )
        if args.include_big_code:
            run_step(
                "bigclonebench",
                statuses,
                lambda: download_hf_named(
                    args,
                    "google/code_x_glue_cc_clone_detection_big_clone_bench",
                    note="C-UDA BigCloneBench clone-detection pairs; JSONL export is capped by --optional_export_limit.",
                    export_limit=args.optional_export_limit,
                ),
            )

    if args.include_leandojo and not args.skip_github:
        run_step(
            "github:LeanDojo-v2",
            statuses,
            lambda: stage_git_repo(
                args,
                "LeanDojo-v2",
                "https://github.com/lean-dojo/LeanDojo-v2.git",
                "LeanDojo-v2 source repository. Install/export proof-progress data with scripts/setup_leandojo_progress.sh.",
            ),
        )

    manifest = {
        "data_dir": str(args.data_dir),
        "profile": args.profile,
        "statuses": [asdict(status) for status in statuses],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
