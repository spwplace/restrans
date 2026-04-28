# Dataset Lake Status

Date: 2026-04-27

The local dataset lake lives under `data/`, which is intentionally gitignored.
The machine-readable manifest is `data/dataset_manifest.json`.

## Current Footprint

```text
data/              34G
data/hf_datasets   22G
data/processed    3.0G
data/source_repos 760M
data/source_archives 183M
```

## Staged Datasets

- WikiText-2 and WikiText-103:
  - `data/hf_datasets/wikitext`
- TinyStories fast subset:
  - `data/hf_datasets/tinystories`
  - currently 100k train / 5k validation rows.
  - Use `--tinystories_full` for a full refresh.
- BLiMP:
  - `data/hf_datasets/blimp`
  - `data/processed/blimp`
  - 67 paradigms, 67k combined rows.
- COGS:
  - `data/hf_datasets/cogs`
  - `data/processed/cogs`
- Public AMR derivative:
  - `data/hf_datasets/amr_3_parsed_public_derivative`
  - `data/processed/amr_3_parsed_public_derivative`
  - This is not a substitute for licensed LDC2020T02 in formal AMR claims.
- BabyLM:
  - strict-small snapshot: `data/hf_datasets/babylm_strict_small_snapshot`
  - strict-small JSONL: `data/processed/babylm_strict_small`
  - full raw snapshot: `data/hf_datasets/babylm_full_snapshot`
- Linzen subject-verb agreement:
  - repo: `data/source_repos/rnn_agreement`
  - simple archive extract: `data/processed/rnn_agreement_simple`
  - dependency TSV: `data/processed/rnn_agreement/agr_50_mostcommon_10K.tsv`
- SLOG:
  - repo: `data/source_repos/slog`
  - generalization sets: `data/processed/slog_generalization_sets`
- HANS:
  - repo/data: `data/source_repos/hans`
- MSGS:
  - repo: `data/source_repos/msgs`
  - extracted data: `data/processed/msgs`
- SCAN:
  - repo/data: `data/source_repos/scan`
- CFQ:
  - `data/hf_datasets/cfq/mcd1`
  - `data/hf_datasets/cfq/mcd2`
  - `data/hf_datasets/cfq/mcd3`
- CodeXGLUE / code datasets:
  - POJ-104: `data/hf_datasets/optional/google__code_x_glue_cc_clone_detection_poj104`
  - CodeSearchNet pair: `data/hf_datasets/optional/sentence-transformers__codesearchnet__pair`
  - BigCloneBench: `data/hf_datasets/optional/google__code_x_glue_cc_clone_detection_big_clone_bench`
- Generic second-wave probes:
  - `data/processed/probes`
  - HANS, MSGS, SLOG, COGS, CFQ, POJ-104, BigCloneBench, and CodeSearchNet
    converted to yes/no JSONL probes.
- External research-code references:
  - cap-matching reference implementation: `external/reu_unif`
  - fetched by `scripts/setup_external_deps.sh`
  - pinned revision and source URL are documented in `external-deps.md`.

## Staging Knobs

Some command-line numbers are resource controls, not experimental constants:

- `--tinystories_train` and `--tinystories_val` choose the fast TinyStories
  subset unless `--tinystories_full` is set.
- `--linzen_style_train` and `--linzen_style_val` choose how many generated
  synthetic Linzen-style rows to write.
- `--optional_export_limit` only caps convenience JSONL mirrors for large
  optional datasets. It does not truncate the saved Hugging Face dataset.
- `--babylm_full` snapshots the complete BabyLM repository. Without it, the
  downloader stages strict-small for fast iteration.

## Current Use

The most promising near-term real tasks are:

- Linzen / BLiMP / MSGS / SLOG for syntax and structural generalization.
- CFQ / COGS for semantic parsing compositionality.
- POJ-104 / BigCloneBench for program equivalence or problem-class structure.
- BabyLM as the controlled natural-language mixture once structural probes are
  wired into training.
- HANS / MSGS / SLOG / COGS / CFQ / POJ-104 / BigCloneBench as second-wave
  generic probes after the first executable wave identifies promising regimes.
- Cap matching / algebraic protocol probes as self-contained synthetic formal
  regimes that explicitly reward knowledge-based unification and constructor
  closure reasoning.
