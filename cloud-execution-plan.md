# Cloud Execution Plan

Date: 2026-04-28

## Acquisition Status

Most pending backlog items are public and actionable. The current blockers are
engineering integration, not licensing.

| Item | Status | Action |
|---|---|---|
| EquiBench | Public project page and Hugging Face dataset. The project page reports 2,400 program pairs across four languages and six equivalence categories. | Acquire immediately and convert to `generic_probe` plus contrastive pair format. |
| CETBench | Public paper; dataset/code location needs confirmation. | Scout after EquiBench. Do not block medium runs. |
| LeanDojo-v2 / LeanProgress | Public toolkit; LeanProgress can export step-prediction JSONL through LeanDojo-v2. Needs GitHub token, optional HF token, Lean/elan setup, and disk. | Stage on a CPU-heavy cloud box first; do not put this on the critical GPU path. |
| ListOps / SCAN | Public-style canaries; local ListOps-style generator already implemented. | Use local generator now; public staging later for comparability. |
| AMR 3.0 / LDC2020T02 | Licensed LDC dataset under the LDC user agreement; not a public-download benchmark. | Defer. Use public AMR derivatives only for development, not formal AMR claims. |
| OGB / CLRS | Public. | Optional; only stage if graph/algorithmic branch becomes central. |
| BabyLM / BLiMP / HANS / MSGS / COGS / SLOG / CFQ | Already staged or public. | Keep in current medium suite. |

### Acquisition Decision

Do not wait for AMR or CETBench before running the main matrix. The current
actionable acquisition queue is:

1. EquiBench, because it is the cleanest public program-equivalence benchmark
   currently identified.
2. LeanDojo-v2 / LeanProgress sample export, because it tests proof-state
   structure but should run on CPU and storage resources first.
3. Public comparability copies of ListOps / SCAN if the local generators show
   signal.
4. OGB / CLRS only if the graph/algorithmic branch becomes a paper axis.

License or access blockers:

- AMR 3.0 / LDC2020T02 needs a proper LDC license before formal use.
- CETBench is not yet blocking because EquiBench covers the code-equivalence
  role with a located public dataset.
- LeanDojo-v2 is an infrastructure setup blocker, not a licensing blocker; it
  needs GitHub API access and a stable Lean toolchain.

## Compute Policy

Small probes are no longer permission gates. They are debugging and
instrumentation checks. The convincing experiment is a medium matrix with:

- matched baselines,
- structurally meaningful tasks,
- multiple seeds,
- stream-specific diagnostics,
- checkpointed artifacts for causal analysis.

## Recommended Cloud Layout

### Controller

Use one cheap CPU instance or the local Mac as controller:

- owns the git checkout,
- launches jobs,
- syncs outputs,
- builds lab notebooks,
- maintains the run manifest.

Do not train from the controller. It should launch jobs, collect results, and
render reports.

### Storage

Use object storage as the source of truth:

- `s3://restrans-runs/datasets/`
- `s3://restrans-runs/checkpoints/`
- `s3://restrans-runs/outputs/`
- `s3://restrans-runs/manifests/`

Each run should write:

- `config.json`,
- `results.json`,
- `report.md`,
- `stdout.log`,
- optional checkpoint,
- optional activation/probe artifacts.

The cloud workers should treat object storage as immutable input plus append-only
outputs. Never rely on a provider-local disk as the only copy of a useful run.

### RunPod

Use RunPod for fast single-node iteration:

- best for 24 GB to 80 GB single-GPU jobs,
- fast to start,
- good for validating the medium matrix on one or two GPUs,
- less ideal as the long-term artifact store.

Target jobs:

- `literature_medium_suite` at current scale,
- MPS/ROCm parity reproduction on CUDA,
- quick 10M-ish BabyLM/TinyStories transfer runs.

Preferred shape:

- RTX 4090 / RTX 5090 class for cheap 5M-10M smoke and many single-seed runs.
- A100 80 GB / H100 80 GB class for 10M-50M narrowed runs, larger batch sizes,
  and activation/probe artifact collection.
- MI300X can be used if the ROCm image is stable, but CUDA should be the first
  cross-cloud reference path because more tooling expects it.

### AWS

Use AWS for sustained runs and heavier sweeps:

- S3 for artifacts,
- EC2 GPU instances for training,
- CPU instances for dataset generation / Lean tracing,
- optional spot where interruption is acceptable,
- on-demand or capacity reservation for final paper runs.

Target jobs:

- full 10M-30M matrix,
- 30M-50M replication on narrowed condition set,
- longer BabyLM structural-transfer run,
- LeanProgress data generation on CPU, then GPU training.

Preferred shape:

- CPU c/m/r-family instance or Batch job for data conversion, Lean tracing, and
  report aggregation.
- G6/L4 for cheap CUDA validation, small probes, and CPU-rich mixed jobs.
- P4d/P4de A100 or P5 H100 for larger final runs when spot/capacity is
  available.
- P6/Blackwell only if we later scale beyond this paper's 10M-50M regime; it is
  not necessary for the current science.

AWS is the safer place for long-running, checkpointed, reproducible paper runs;
RunPod is the faster place for interactive GPU availability and throughput
checks.

## Execution Stages

### Stage 0: Cloud Image Validation

Goal: prove the repo runs cleanly on one fresh cloud machine.

Commands:

```bash
git clone <repo>
cd restrans
uv sync
bash scripts/setup_external_deps.sh
PYTHONPATH=resonance uv run python resonance/tests/test_smoke.py
DRY_RUN=1 bash scripts/run_literature_medium_suite.sh
```

Exit criteria:

- smoke tests pass,
- dry-run manifest is sensible,
- outputs sync to object storage.

### Stage 1: Data Acquisition And Conversion

Run on CPU, not the expensive GPU.

Primary command:

```bash
bash scripts/acquire_research_assets.sh
```

Fuller cloud acquisition:

```bash
BABYLM_FULL=1 INCLUDE_BIG_CODE=1 INCLUDE_CODESEARCHNET=1 INCLUDE_LEANDOJO=1 \
  bash scripts/acquire_research_assets.sh
```

Acquire now:

- EquiBench HF dataset,
- LeanDojo-v2 source,
- public ListOps/SCAN if desired,
- any missing HF mirrors for already-selected tasks.

Do not wait for:

- AMR 3.0 LDC.
- CETBench location confirmation.

Exit criteria:

- `dataset_manifest.json` updated,
- EquiBench converted to binary pair probe and contrastive pair files,
- LeanProgress feasibility documented with one sample JSONL export.

Suggested CPU jobs:

```text
stage1_equibench_convert_cpu
stage1_leandojo_trace_sample_cpu
stage1_dataset_manifest_cpu
```

LeanProgress sample export:

```bash
export GITHUB_ACCESS_TOKEN=<token>
bash scripts/setup_leandojo_progress.sh
```

### Stage 2: Medium Matrix Smoke On RunPod

Run the current medium suite on a single CUDA GPU.

Use reduced but nontrivial settings:

```bash
DEVICE=cuda \
SAVE_MODELS=1 \
RUN_INTERP=1 \
EPOCHS=4 \
TRAIN_EXAMPLES=1500 \
VAL_EXAMPLES=700 \
EMBED_DIM=128 \
LAYERS=4 \
HEADS=4 \
FF_DIM=512 \
BATCH_SIZE=24 \
bash scripts/run_literature_medium_suite.sh
```

Purpose:

- find runtime failures,
- estimate throughput,
- identify obviously saturated/impossible tasks,
- check whether ALiBi/DeBERTa-lite dominate.

Exit criteria:

- all tasks complete,
- lab notebook builds,
- at least two task families are learnable but not saturated.

### Stage 3: Full 10M-ish Matrix

Run on RunPod A100/H100 or AWS G/P family depending on availability.

Conditions:

- `standard`
- `standard_iso`
- `standard_alibi`
- `standard_deberta_lite`
- `phase_stream_only_normalized`
- `phase_stream_only_normalized_phase_contrastive`
- `resonance_full_normalized`
- `phase_dynamic_qk_film`
- `complex_directional_normalized`
- `relational_stream_lite`

Tasks:

- cap matching,
- unification,
- algebraic protocol,
- ListOps-style,
- SLOG,
- COGS/CFQ,
- HANS/MSGS diagnostics,
- EquiBench if conversion is ready.

Seeds:

- minimum 3,
- 5 for the top two task families after first pass.

Exit criteria:

- result table with confidence intervals,
- phase-vs-residual probes for selected tasks,
- ablation sensitivity on winning conditions.

Suggested split:

- RunPod: two to four short task-family jobs to identify breakage and obvious
  winners.
- AWS: the full seed matrix, because it is easier to checkpoint, resume, and
  preserve artifacts in S3.
- Local Mac / persvati: keep running CPU-heavy synthetic data generation,
  analysis, and report rendering so cloud GPUs do not idle waiting for plots.

### Stage 4: Narrowed 30M-50M Replication

Only after Stage 3.

Keep at most 4-5 conditions:

- best standard baseline,
- best existing-literature baseline,
- best structural-stream variant,
- best structural-stream + pressure variant,
- legacy resonance only if still competitive.

Run on:

- SLOG/CFQ or COGS,
- EquiBench or formal synthetic suite,
- BabyLM transfer if structural runs justify it.

Exit criteria:

- clean plots,
- causal interpretation bundle,
- decision on whether this is a paper, negative result, or redesign.

## Parallelization

Preferred sharding:

- shard by task first,
- then by condition,
- then by seed.

Avoid mixing tasks in one long process on cloud. One task-family per job makes
failures cheap and artifacts easier to reason about.

Suggested job naming:

```text
{date}_{stage}_{task}_{scale}_{condition_set}_{seed_set}
```

Example:

```text
2026-04-28_stage3_slog_10m_litmatrix_seeds701-707
```

Concrete first launch batch:

```text
cpu-1: acquire/convert EquiBench, update manifest
cpu-2: LeanDojo-v2 install + sample LeanProgress export
gpu-1: cap_matching + unification + algebraic_protocol, all conditions, 3 seeds
gpu-2: listops + dyck, all conditions, 3 seeds
gpu-3: SLOG + COGS, all conditions, 3 seeds
gpu-4: CFQ + EquiBench + HANS + MSGS, all conditions, 3 seeds
```

After that batch, stop sweeping all variants everywhere. Pick the top two task
families and the top four conditions, then spend compute on bigger models,
more seeds, ablation, patching, and probes.

## Minimum Convincing Result

A convincing first result is:

1. at least two structure-heavy task families show selective improvement,
2. improvement survives `standard_iso`, `standard_alibi`,
   `standard_deberta_lite`, and `relational_stream_lite`,
3. phase/structural states are more probeable or compressible than residual
   states for task structure,
4. stream ablation or patching selectively damages structural decisions.

If only one of those holds, scale cautiously. If none hold, stop or redesign.

## External References Checked

- EquiBench project page: https://anjiang-wei.github.io/EquiBench-Website/
- EquiBench Hugging Face dataset:
  https://huggingface.co/datasets/anjiangwei/EquiBench-Datasets
- LeanProgress: https://leandojo.org/leanprogress.html
- LeanDojo-v2 PyPI docs: https://pypi.org/project/lean-dojo-v2/
- AMR 3.0 / LDC2020T02: https://catalog.ldc.upenn.edu/LDC2020T02
- AWS P5/P6/G6 instance pages:
  https://aws.amazon.com/ec2/instance-types/p5/
  https://aws.amazon.com/ec2/instance-types/p6/
  https://aws.amazon.com/ec2/instance-types/g6/
- RunPod GPU types: https://docs.runpod.io/references/gpu-types
