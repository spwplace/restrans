# Active Experiment Log

Updated: 2026-04-28T22:55:18Z

This file tracks the live execution state so the repo is handoff-readable even
while nextop and persvati are running.

## 2026-04-30 15:05 UTC — Step-to-Trace Curriculum Pivot

Current read: a one-step verifier should become meaningful by composing into a
multi-step trace verifier. The new posttraining harness now supports explicit
SFT curriculum tasks before the target RL/DPO task.

Implemented:

- `lambda_beta_step -> lambda_trace`: learn exact normal-order beta transition,
  then train on whole beta-reduction cascades.
- `vm_step -> vm_trace`: learn one stack-machine instruction transition, then
  train on whole execution traces.
- Trace examples now include dense verifier metadata such as valid local steps,
  terminal validity, and `process_score`. The binary answer objective still uses
  yes/no labels, but the data now exposes process supervision for the next
  generated-trace RL stage.
- `dfa_equivalence` is available as an exact finite-automata equivalence task.

Smoke tests passed locally on CPU for both curriculum paths:

```bash
PYTHONPATH=resonance uv run python resonance/train_structural_posttrain.py \
  --task lambda_trace --curriculum_tasks lambda_beta_step \
  --curriculum_epochs 1 --sft_epochs 1 --rl_epochs 1 --rl_algorithm dpo \
  --conditions standard_alibi --train_examples 24 --val_examples 16 \
  --curriculum_train_examples 24 --device cpu

PYTHONPATH=resonance uv run python resonance/train_structural_posttrain.py \
  --task vm_trace --curriculum_tasks vm_step \
  --curriculum_epochs 1 --sft_epochs 1 --rl_epochs 1 --rl_algorithm exact_rl \
  --conditions relation_value_qk_film_alibi_normalized \
  --train_examples 24 --val_examples 16 --curriculum_train_examples 24 \
  --device cpu
```

Active cascade runs:

- nextop: `resonance/outputs/posttraining_cascade_nextop_2026_04_30_144727`
  via tmux session `restrans_posttraining_cascade_nextop_2026_04_30_144727`.
  Runs DPO on MPS with `lambda_beta_step -> lambda_trace` and
  `vm_step -> vm_trace` curricula.
- persvati: `resonance/outputs/posttraining_cascade_persvati_2026_04_30_144727`
  via tmux session `restrans_posttraining_cascade_persvati_2026_04_30_144727`.
  Runs exact-RL and GRPO-enum on ROCm with the same curricula.

## 2026-04-30 19:40 UTC — Cascade Runs Complete, Persvati Space Recovered

Both cascade runs finished.

Local nextop output:

- `resonance/outputs/posttraining_cascade_nextop_2026_04_30_144727`
- Algorithm: DPO
- Tasks: `lambda_trace`, `vm_trace`, `dfa_equivalence`, `cap_matching`,
  `algebraic_protocol`

Remote persvati output, synced locally:

- `resonance/outputs/posttraining_cascade_persvati_2026_04_30_144727`
- Algorithms: exact-RL and GRPO-enum
- Tasks: same five-task suite

Combined status artifacts:

- `lab_notebook_hyper_live/cascade_posttraining_report.md`
- `lab_notebook_hyper_live/cascade_posttraining_combined.csv`
- `lab_notebook_hyper_live/plots/cascade_posttraining_accuracy.png`
- `lab_notebook_hyper_live/plots/cascade_composition_accuracy.png`
- `lab_notebook_hyper_live/plots/phase_permutation_delta.png`

Result read:

- No universal architecture win. `standard_alibi` remains the strongest direct
  answer classifier on many tasks.
- `phase_dynamic_qk_film_alibi_normalized` shows strong phase geometry and
  several causal phase-permutation hits under verifier training, especially
  exact-RL / GRPO settings.
- `relation_value_qk_film_alibi_normalized` is most interesting as a
  one-step-to-trace composition mechanism: it improves cascade accuracy on
  nextop lambda and VM traces, and on persvati exact-RL VM trace, even when
  direct yes/no accuracy is not best.
- VM trace remains close to chance as a direct classifier. It should become a
  process-reward / generated-trace task, not just a final answer-token task.
- DFA equivalence is suspicious because several conditions converge to the
  same final accuracy; inspect the generator and decision modes before using it
  as evidence.

Persvati disk cleanup:

- Problem source: `/home/ember/.cache/huggingface` held about 1.2 TB of public
  dataset cache, mostly FineWeb-EDU, C4, and OpenWebText blobs/Arrow caches.
- Removed redownloadable HF cache directories, not repo outputs or source code.
  This was too aggressive because those global caches were still part of the
  working web-corpus lake for the webscale experiments.
- Disk recovered from about 99% used / 27 GB free to about 33% used / 1.2 TB
  free.

Recovery / guardrails:

- Repo-local datasets in `restrans-exp/data` were intact after cleanup:
  BabyLM, TinyStories, BLiMP, COGS, CFQ, EquiBench, CodeSearchNet, POJ,
  BigCloneBench, AMR derivative, source repos, and processed probes.
- The damaged part is the global FineWeb/OpenWebText/C4 HF cache. Nextop still
  has `data/hf_datasets/HuggingFaceFW___fineweb-edu/sample-10BT` (~46 GB).
- `resonance/train_webscale.py` now defaults to repo-scoped
  `data/hf_home` / `data/hf_datasets`, passes `cache_dir` explicitly, and uses
  non-streaming prepared cache unless `--streaming` is supplied.
- `scripts/download_datasets.py` no longer hardcodes `~/dev/restrans`; it uses
  the current repo root / `DATA_DIR`.
- Restore command to resume when persvati is reachable:

```bash
rsync -aW --partial --info=progress2 \
  -e "ssh -i ~/.ssh/id_aws -o ConnectTimeout=10" \
  data/hf_datasets/HuggingFaceFW___fineweb-edu/ \
  persvati.local:/home/ember/restrans-exp/data/hf_datasets/HuggingFaceFW___fineweb-edu/
```

- Alternative direct rehydrate on persvati:

```bash
cd /home/ember/restrans-exp
source .venv-exp/bin/activate
export UV_PROJECT_ENVIRONMENT=.venv-exp
DATA_DIR=data uv run python scripts/download_datasets.py
```

## Current Research Read

- The original broad claim has narrowed into a testable one: an explicit
  compact structural stream may help when the task contains controlled
  relational invariants and the model is forced to use them.
- We should not treat pairwise attention bias, phase coordinates, or cosine
  differences as novel by themselves. The relevant comparisons are ALiBi,
  DeBERTa-style splits, relation-aware attention, and Abstractor/DAT-like
  relational streams.
- The most informative evidence now comes from three simultaneous checks:
  selective task gains, cleaner structural geometry, and causal/intervention
  sensitivity. Accuracy-only bumps are scouting signal.

## Completed Since Last Handoff

- Stopped the old nextop matrix during `listops_depth4`; ListOps was slow and
  not discriminating the structural claim.
- Preserved and summarized the completed nextop partial matrix:
  `resonance/outputs/architecture_matrix_nextop_2026_04_28_medium_mps`.
- Added a focused local launcher:
  `scripts/run_signal_matrix_nextop.sh`.
- Added literature-inspired variants:
  `harmonic_cosine_normalized`, `relation_value_mix_normalized`,
  `relation_value_qk_film_normalized`, and `harmonic_relation_value_normalized`.
- Added `resonance/plot_matrix_results.py` for per-run heatmaps, matrix CSVs,
  and compact reports.
- Extended smoke coverage for the harmonic kernel and relation-value routing.
- Added a pair-specific MLP relation kernel for future sweeps:
  `pair_mlp_kernel_normalized` and `pair_mlp_relation_value_normalized`.

## Persvati Second-Wave Results

Existing completed directory:
`resonance/outputs/hyper_second_wave_2026_04_28`.

Read:

- `cogs_gen_semantic_parse_gpu` is a candidate regime. It is learnable and
  non-saturated; structural variants were ahead of standard in this setup.
- `slog_gen_cogs_lf_gpu` is too weak as configured. It barely trains and should
  not be used as evidence until the task/data presentation is fixed.
- `poj104_problem_id_gpu` is not in a meaningful regime. Treat it as failed
  regime selection, not an architecture result.
- `bigclonebench_clone_probe_gpu` failed because `max_length=1024` was too
  short for examples. Rerun only with higher max length and small batch size.

## Active Runs

### nextop

Root:
`resonance/outputs/signal_matrix_nextop_2026_04_28_mps`

Command:

```bash
ROOT=resonance/outputs/signal_matrix_nextop_2026_04_28_mps \
DEVICE=mps SEEDS="941 943 947" EPOCHS=4 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
bash scripts/run_signal_matrix_nextop.sh
```

Task set:

- `unification_depth5`
- `cap_matching_depth4`
- `algebraic_protocol_depth4`
- `graph_alias_new`
- `cogs_gen_semantic_parse`
- `equibench_oj_va`

Partial read after `unification_depth5` and `cap_matching_depth4`:

- Both tasks are learnable and non-saturated.
- `unification_depth5` favors `standard` and `standard_iso_deberta_lite`.
- `cap_matching_depth4` is closer: `phase_dynamic_qk_film_normalized`,
  `standard_deberta_lite`, and `standard` are effectively tied at the top.
- Full resonance, phase-only, harmonic cosine, and relation-value mix currently
  track each other closely. This is evidence against the original additive
  relation-bias mechanism as a standalone win in these regimes.

### persvati

Root:
`resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2`

Command shape:

```bash
source .venv-exp/bin/activate
export UV_PROJECT_ENVIRONMENT=.venv-exp
export HSA_OVERRIDE_GFX_VERSION=11.0.0
ROOT=resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2 \
DEVICE=cuda SEEDS="951 953" EPOCHS=5 \
TRAIN_EXAMPLES=2500 VAL_EXAMPLES=800 \
HARD_TRAIN_EXAMPLES=1800 HARD_VAL_EXAMPLES=600 \
COGS_TRAIN_EXAMPLES=3000 COGS_VAL_EXAMPLES=1000 \
CODE_TRAIN_EXAMPLES=500 CODE_VAL_EXAMPLES=160 \
EMBED_DIM=128 LAYERS=4 HEADS=4 FF_DIM=512 \
BATCH_SIZE=16 COGS_BATCH_SIZE=16 CODE_BATCH_SIZE=2 \
bash scripts/run_signal_matrix_nextop.sh
```

Note: persvati's ROCm torch is installed in `.venv-exp`. Plain `uv run` without
`UV_PROJECT_ENVIRONMENT=.venv-exp` does not see torch there.

## Post-Run Commands

For each completed root:

```bash
uv run python resonance/summarize_architecture_matrix.py \
  --root <root> \
  --output_json <root>/summary.json \
  --output_md <root>/summary.md

uv run python resonance/plot_matrix_results.py --root <root>
```

Then rebuild the global notebook:

```bash
uv run python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook_hyper_live
```

If persvati results are not local, sync them:

```bash
rsync -az -e "ssh -i ~/.ssh/id_aws" \
  persvati:restrans-exp/resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2/ \
  resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2/
```

## 2026-04-28 19:30 UTC — Handoff from GPT-5.5 to Kimi Code CLI

Both runs healthy.

### Local nextop (signal_matrix_nextop_2026_04_28_mps)
- Completed: unification_depth5, cap_matching_depth4, algebraic_protocol_depth4
- Running: graph_alias_new (seed 943, ~10 min in)
- Pending: cogs_gen_semantic_parse, conditional equibench_oj_va
- 12 conditions x 3 seeds, 64d/2-layer, 4 epochs
- Key pattern so far: phase_dynamic_qk_film is the only structural variant that consistently competes with strong baselines (standard_iso_deberta_lite, standard). Full resonance and phase_stream_only are nearly identical across all three tasks, reinforcing that static additive relation bias is not the active ingredient.

### Remote persvati (signal_matrix_persvati_2026_04_28_rocm_v2)
- Still on unification_depth5 seed 953 (harmonic_cosine_normalized epoch 2 as of last check)
- Seed 951 completed all conditions for unification_depth5 but results.json not yet written (probe writes per-task after all seeds finish)
- 12 conditions x 2 seeds, 128d/4-layer, 5 epochs, larger data
- ROCm environment: `.venv-exp` with `UV_PROJECT_ENVIRONMENT=.venv-exp`

### Code state
- Committed: all new runners, kernels (harmonic, pair_mlp, relation-value), plotting/summarization tools, smoke tests, app research-site updates
- Branches: dev (all work on dev branch)

### Commands to run when local finishes
```bash
# Already auto-runs at end of script, but if manually needed:
uv run python resonance/summarize_architecture_matrix.py \
  --root resonance/outputs/signal_matrix_nextop_2026_04_28_mps \
  --output_json resonance/outputs/signal_matrix_nextop_2026_04_28_mps/summary.json \
  --output_md resonance/outputs/signal_matrix_nextop_2026_04_28_mps/summary.md

uv run python resonance/plot_matrix_results.py \
  --root resonance/outputs/signal_matrix_nextop_2026_04_28_mps \
  --output_dir resonance/outputs/signal_matrix_nextop_2026_04_28_mps/matrix_plots
```

### Commands to sync persvati when ready
```bash
rsync -az -e "ssh -i ~/.ssh/id_aws" --exclude '*.pt' --exclude 'checkpoints' \
  persvati:restrans-exp/resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2/ \
  resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2/
```

## 2026-04-28 19:40 UTC — Launched persvati CPU LM focused run

Target: test `phase_dynamic_qk_film_normalized` against strong baselines on language modeling.

Command:
```bash
ssh persvati
cd restrans-exp
source .venv-exp/bin/activate
export UV_PROJECT_ENVIRONMENT=.venv-exp OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
bash scripts/run_cpu_lm_focused.sh
```

- Root: `resonance/outputs/cpu_lm_focused_2026_04_28_234016`
- Device: cpu (8 threads, leaving GPU run's 12 threads alone)
- Conditions: `standard,standard_alibi,standard_iso_deberta_lite,phase_dynamic_qk_film_normalized`
- Seeds: 123, 456
- Datasets: BabyLM strict small, TinyStories
- Model: 128d, 3 layers, 4 heads, 512 ff_dim, seq_len 96, batch 32
- Epochs: 3
- Train/val chunks: BabyLM 6000/800, TinyStories 8000/1000

Status: Training BabyLM seed=123, condition `standard`, ~batch 100/188.

## 2026-04-28 19:45 UTC — Launched local nextop CPU LM runs

Nextop CPU is participating while MPS runs on GPU.

### Jobs launched
1. **TinyStories seed 789** — `resonance/outputs/cpu_lm_nextop_tinystories_2026_04_28/seed789`
2. **TinyStories seed 791** — `resonance/outputs/cpu_lm_nextop_tinystories_2026_04_28/seed791`
3. **BabyLM seed 789** — `resonance/outputs/cpu_lm_nextop_babylm_2026_04_28/seed789`

All use:
- Conditions: `standard,standard_alibi,standard_iso_deberta_lite,phase_dynamic_qk_film_normalized`
- Model: 128d, 3 layers, 4 heads, 512 ff_dim, seq_len 96, batch 24
- Epochs: 3
- Device: cpu with `OMP_NUM_THREADS=6` each

### Early signal
- TinyStories seed 789, `standard`: Val PPL 47.75 after 3 epochs (very fast, ~0.8 min/epoch)
- BabyLM seed 789, `standard`: epoch 1 in progress

## 2026-04-28 20:00 UTC — Nextop CPU saturated, app rebuilt

Local CPU now running 5 concurrent LM jobs:
- TinyStories seeds 789, 791, 793
- BabyLM seeds 789, 791
- All: 128d/3-layer, 3 epochs, 4 conditions, OMP_NUM_THREADS=6 each

App updated and built:
- Signal matrix table (nextop 64d/2-layer, 3 tasks)
- Persvati unification table (128d/4-layer, 2 seeds)
- Capacity caution panel (param mismatches, seed limits, replication needed)
- LM run status panel
- All pages prerendered successfully

## 2026-04-28 21:00 UTC — Launched unified overnight runs (Path A)

Stopped all small architecture sweeps. Killed signal matrices on both nextop and persvati.
Launched unified mixed trainer that interleaves LM + structural tasks.

### Nextop M2 Max (MPS)
- Root: `resonance/outputs/unified_nextop_2026_04_28`
- Model: 384d / 8 layers / 8 heads / 1536 ff_dim = **20.4M params**
- Data: BabyLM (80k chunks) + TinyStories + unification(5k) + cap_matching(5k) + algebraic_protocol(5k)
- Batch: 48, Seq len: 128, Epochs: 5
- Conditions: `standard_alibi`, `phase_dynamic_qk_film_normalized`
- Seeds: 789, 791
- Mix: 2 LM batches per 1 task batch

### Persvati APU (ROCm)
- Root: `resonance/outputs/unified_persvati_2026_04_28`
- Model: 256d / 6 layers / 8 heads / 1024 ff_dim = **8.9M params**
- Data: BabyLM (40k chunks) + TinyStories + unification(3k) + cap_matching(3k) + algebraic_protocol(3k)
- Batch: 64, Seq len: 128, Epochs: 5
- Conditions: `standard_alibi`, `phase_dynamic_qk_film_normalized`
- Seeds: 123
- Mix: 2 LM batches per 1 task batch

### What this tests
Whether training on language modeling + structural reasoning together produces a model where
`phase_dynamic_qk_film` generalizes better than `standard_alibi` on held-out LM and task eval.
This is the "big pot" the user asked for.

### Expected finish
- Nextop: ~15-20 hours (late morning tomorrow)
- Persvati: ~12-16 hours (midday tomorrow)

## 2026-04-30 14:30 UTC — Pivoted to structural posttraining / RLVR

Stopped treating web-scale LM as the primary discovery loop. Added a verifier-centered
posttraining harness and new exact task families:

- `lambda_equivalence`: beta-normal-form equivalence.
- `lambda_beta_step`: one-step normal-order beta validity.
- `lambda_trace`: whole beta-reduction cascade validity.
- `vm_trace`: stack-machine execution trace validity.
- `vm_equivalence`: same final stack for two VM programs.
- `dfa_equivalence`: accepted-language equivalence under state renaming.

New runner:

```bash
DEVICE=mps ALGORITHMS=dpo bash scripts/run_structural_posttraining.sh
```

New training algorithms in `resonance/train_structural_posttrain.py`:

- SFT warm start.
- DPO over correct vs incorrect answer token.
- Exact expected-reward RL for binary verifier rewards.
- Enumerated two-answer GRPO-style update.

### Active runs

Nextop:

- tmux: `restrans_posttrain_nextop_142354`
- root: `resonance/outputs/posttraining_pilot_nextop_tmux_2026_04_30_142354`
- algorithm: DPO
- tasks: lambda trace, VM trace, cap matching, algebraic protocol
- model: 96d / 3 layers / 4 heads / 384 ff_dim

Persvati:

- tmux: `restrans_posttrain_persvati_142902`
- root: `resonance/outputs/posttraining_pilot_persvati_2026_04_30_142902`
- algorithms: exact-RL and enumerated-GRPO
- tasks: lambda trace, VM trace, DFA equivalence, cap matching, algebraic protocol
- model: 128d / 4 layers / 4 heads / 512 ff_dim
- required env: `HSA_OVERRIDE_GFX_VERSION=11.0.0`

### Notes

The first persvati launch failed with ROCm `invalid device function`; this matched the existing
ROCm setup issue and was fixed by reusing the older `HSA_OVERRIDE_GFX_VERSION=11.0.0` environment.

The first lambda-trace generator used rejection sampling for reducible terms and was too slow for
real runs. It now constructs guaranteed multi-step beta-redex cascades directly.
