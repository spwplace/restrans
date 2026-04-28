# HYPEREXPERIMENT: Phase-Structural Resonance Variants

Date: 2026-04-28

Literature integration note: the closest prior art includes relation-aware
attention, graph transformer structural biases, Abstractor-style relational
bottlenecks, and Dual Attention Transformer-style sensory/relational splits.
Therefore, this document should be read as a model-variant workbench, not a
novelty claim. The active project policy is in
`research/literature_integration.md`.

This is the handoff document for the next research agent. Read this before
touching code. The project is **not** currently trying to prove a vague
"resonance is magic" story. The sharpened question is:

> Can a compact learned phase/structural channel give a transformer a useful
> inductive bias for structural equivalence, validity, and topology-bearing
> natural language/program tasks?

The current evidence says: **maybe yes for the phase/structural channel**,
especially on BLiMP `wh_island` low-data regimes. The evidence does **not** yet
show that the current scalar additive attention bias is the important mechanism.

## Current Takeaway

The current model has two separable ideas:

1. **Phase stream / structural channel**: token embeddings include a compact
   low-dimensional phase representation projected/blended into the model.
2. **Scalar additive relation bias**: pairwise phase similarity is reduced to
   one scalar matrix and added to attention logits.

The first idea is still central. The second may be a weak or poorly designed
readout.

Do not overclaim. The most honest current claim is:

> Phase-bearing models show promising low-data structural syntax behavior, but
> full resonance and phase-only are often tied. Therefore, the phase/structural
> representation is the current object of study; the present additive
> attention-bias adapter is not yet justified as the key mechanism.

## Most Important Results So Far

Remote `persvati` runs are still active at time of writing, but completed
results include:

### BLiMP `wh_island_n150`

This is the best current positive signal.

| condition | acc | acc-majority | loss gain | label gap |
|---|---:|---:|---:|---:|
| `standard` | 0.7113 | +0.2113 | -0.0358 | 0.6851 |
| `resonance_full_normalized` | 0.7913 | +0.2913 | +0.2126 | 0.8882 |
| `phase_stream_only_normalized` | 0.7927 | +0.2927 | +0.2112 | 0.8844 |
| `bias_only_normalized` | 0.7593 | +0.2593 | +0.1259 | 0.7854 |
| `resonance_inert_normalized` | 0.7620 | +0.2620 | +0.1299 | 0.7867 |

Interpretation: phase/full are best, but full and phase-only are tied. Bias and
inert also improve. This supports an additional structural channel, not
specifically the current scalar attention bias.

### BLiMP `wh_island_n100`

Same pattern, weaker:

| condition | acc | loss gain | label gap |
|---|---:|---:|---:|
| `standard` | 0.5653 | -0.1606 | -0.2364 |
| `resonance_full_normalized` | 0.6587 | +0.0062 | 0.0264 |
| `phase_stream_only_normalized` | 0.6600 | +0.0110 | 0.0353 |
| `bias_only_normalized` | 0.6013 | -0.1570 | -0.0191 |
| `resonance_inert_normalized` | 0.5993 | -0.1572 | -0.0182 |

### Negatives / Guardrails

- Many BLiMP island/binding probes at `n=100` stay near chance.
- HANS saturates trivially under the current adapter and is not useful.
- MSGS `main_verb_length` currently does not favor the phase models.
- Local synthetic `template_equivalence_depth5` is learnable but favors
  bias/inert, not phase/full.
- Local `graph_alias_new_graphs_small` gives an accuracy bump for full
  resonance but bad loss/geometry; scouting signal only.

## Running Machine Notes

### Local Mac

Use `uv`. MPS is available, but medium MPS jobs died silently after model setup
in this environment. Tiny MPS smoke works. Treat MPS as experimental; do not
depend on it for unattended medium runs.

### `persvati`

SSH:

```bash
ssh -i ~/.ssh/id_aws persvati
```

Project path:

```bash
cd ~/restrans-exp
. .venv-exp/bin/activate
export PYTHONPATH=resonance
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

ROCm PyTorch exposes the AMD GPU through the PyTorch `cuda` device API. Use:

```bash
--device cuda
```

Check active jobs:

```bash
pgrep -af "structural_task_probe|train_corpus_lm|run_overnight_persvati|run_post_persvati"
```

Do not kill running jobs unless necessary. Pull remote outputs with:

```bash
rsync -avz --partial -e 'ssh -i ~/.ssh/id_aws' \
  persvati:restrans-exp/resonance/outputs/ \
  resonance/outputs/
```

Rebuild notebook:

```bash
uv run python resonance/build_lab_notebook.py \
  --root resonance/outputs \
  --output_dir resonance/outputs/lab_notebook
```

Main report target:

```text
resonance/outputs/lab_notebook/lab-notebook.md
```

## Design Diagnosis

The current attention adapter is:

```text
score_ij,h = q_i,h @ k_j,h / sqrt(d) + w_h * mean_f cos(phi_i,f - phi_j,f)
```

This can fail for several reasons:

- It compresses the entire phase relation to one scalar per pair.
- It is static from token embeddings, not a layerwise structural state.
- It is shared too bluntly across roles; relation type and direction are lost.
- It only nudges attention logits after Q/K have already been computed.
- Additive logit bias can be ignored, swamped, or collapse to a capacity knob.
- It does not make the representations themselves structurally aware.

The next model work should preserve the phase/structural-channel idea while
testing better ways for phase geometry to participate in computation.

## Model Variants To Try

Implement these behind config flags and named conditions. Keep each variant
ablated and comparable.

### Variant A: Dynamic Phase State

Make phase evolve per layer instead of remaining a static token embedding.

Current flow:

```text
token_ids -> phase embedding phi_0
same phi_0 used in every layer
```

Target flow:

```text
x_0, phi_0 = embedding(token_ids)
for layer l:
    x_{l+1}, phi_{l+1} = block_l(x_l, phi_l)
```

Minimal implementation:

```text
phi_msg = PhaseAttention(LN_phi(phi_l), optional semantic context x_l)
phi_l = phi_l + phi_msg
phi_l = phi_l + PhaseMLP(LN_phi(phi_l))
R_l = kernel(phi_l)
x_l = semantic attention using R_l or phase-conditioned Q/K
```

Key choices:

- `phase_update_mode = none | mlp | self_attn | cross_attn`
- Keep `n_frequencies` small, e.g. 16/32/64.
- Add a phase residual scale initialized small, e.g. `phase_update_scale=0.1`.
- Log phase drift and effective rank per layer.

Why this matters:

The phase stream becomes a **structural scratchpad**, not just a static tag.
This is likely the highest-priority architecture change.

### Variant B: Phase-Conditioned Q/K

Instead of adding phase relation after QK, use phase to transform Q/K before
attention matching.

FiLM version:

```text
gamma_q, beta_q = MLP_q(phi_i)
gamma_k, beta_k = MLP_k(phi_j)
q'_i = q_i * (1 + gamma_q) + beta_q
k'_j = k_j * (1 + gamma_k) + beta_k
score_ij = q'_i @ k'_j / sqrt(d)
```

Rotation version:

```text
theta_i,h = W_h phi_i
q'_i,h = rotate_pairs(q_i,h, theta_i,h)
k'_j,h = rotate_pairs(k_j,h, theta_j,h)
```

Why this matters:

Phase becomes part of the actual addressing/matching mechanism, rather than an
external logit nudge.

### Variant C: Directional Complex Relation Kernel

Replace mean cosine with a learned directional kernel:

```text
delta_ij = phi_i - phi_j
R_ij,h = a_h · cos(W_h delta_ij) + b_h · sin(W_h delta_ij)
```

Cosine captures symmetric alignment; sine captures direction/order. This is a
better fit for syntax, proof steps, causal edges, and program flow.

Conditions:

- `complex_relation_bias`
- `complex_relation_qk`
- `complex_relation_values`

### Variant D: Dedicated Structural Heads

Reserve some heads for semantic attention and some for phase/structure.

```text
semantic heads: QK over x
structural heads: QK/R over phi
cross heads: x queries phi, or phi queries x
```

Do not blend all signals immediately. The point is disentanglement.

Suggested config:

```text
n_semantic_heads = n_heads - n_structural_heads
n_structural_heads = 1 or 2 initially
```

### Variant E: Relation-Conditioned Values

Do not only change where attention looks; change what information is read based
on the relation.

```text
r_ij = relation_mlp(phi_i, phi_j, phi_i - phi_j)
v'_j_for_i = v_j + W_v r_ij
out_i = sum_j attn_ij * v'_j_for_i
```

This is more expensive. Use only after A/B/C have smoke-tested.

### Variant F: Direct Topological Phase Loss

Do not rely on next-token or yes/no probe loss to discover structure.

Add auxiliary losses:

```text
same-structure positives: phase_summary(a) close to phase_summary(b)
hard negatives: phase_summary(a) far from phase_summary(c)
mutation invariance: phase_summary(program) stable under semantic-preserving edits
```

Tasks with ground-truth structure:

- BLiMP minimal pairs: phase should separate grammatical/ungrammatical.
- Graph alias: same latent graph state should cluster.
- Template equivalence: same template should cluster.
- COGS/SLOG/CFQ: same semantic form or compositional template should cluster.
- Program clone/code problem ID: same semantic problem should cluster.

Use this as a training objective on the phase path, not just as a diagnostic.

## Required Controls

For every serious run, compare:

- `standard`
- `resonance_full_normalized` (current model)
- `phase_stream_only_normalized`
- `bias_only_normalized`
- `resonance_inert_normalized`
- **iso-parameter standard**: standard transformer widened to match phase model
  parameter count
- new variant conditions, one at a time

The iso-parameter baseline is important. Some "phase" gains may simply be extra
capacity unless controlled.

## Code Targets

Primary model files:

```text
resonance/resonance/config.py
resonance/resonance/models.py
resonance/resonance/kernels.py
resonance/resonance/bias_modes.py
```

Experiment switchboard:

```text
resonance/regime_probe.py
resonance/structural_task_probe.py
resonance/run_experiment_plan.py
resonance/summarize_experiment_plan.py
resonance/build_lab_notebook.py
```

Dataset/task generators:

```text
resonance/synthetic/
data/processed/blimp/
data/processed/probes/
```

Current model structure:

- `ResonanceEmbedding` creates semantic embeddings, phase embeddings,
  `phase_proj`, blend gate, and static resonance matrix.
- `ResonanceAttention` currently implements scalar additive resonance bias.
- `ResonanceBlock` passes the same static `phases` through each block.
- `ResonanceTransformer.forward` computes `phases =
  self.embedding.get_phases(input_ids)` once and reuses them.

This is where dynamic phase should be inserted.

## Suggested Implementation Order

Implementation status in this workspace:

- `phase_update_mode` config flags are implemented for `none`, `mlp`, and
  `self_attn`.
- `phase_condition_qk="film"` is implemented with identity initialization.
- `directional_complex` kernel is implemented.
- `n_structural_heads` is implemented as phase-structure-only attention heads.
- `standard_iso` is implemented as a widened standard baseline approximation.
- Smoke tests have passed through `structural_task_probe.py` for:
  - `phase_dynamic_mlp`
  - `phase_dynamic_attn`
  - `phase_qk_film`
  - `phase_dynamic_qk_film`
  - `complex_directional`
  - `structural_heads_1`
  - `standard_iso`

### Step 1: Add Config Flags

In `resonance/resonance/config.py`, add fields:

```python
phase_update_mode: str = "none"          # none | mlp | self_attn
phase_update_scale: float = 0.1
phase_condition_qk: str = "none"         # none | film | rotate
relation_kernel_mode: str = "scalar_cos" # scalar_cos | directional_complex
n_structural_heads: int = 0
relation_value_mode: str = "none"        # none | additive
aux_phase_loss_weight: float = 0.0
```

Keep defaults equivalent to the current model.

### Step 2: Implement Dynamic Phase State

Add a small `PhaseUpdateBlock` in `models.py`:

```python
class PhaseUpdateBlock(nn.Module):
    def __init__(self, config):
        ...
    def forward(self, phase, x=None, mask=None):
        ...
```

Start with `phase_update_mode="mlp"`:

```text
phase = phase + scale * MLP(LayerNorm(phase))
```

Then `phase_update_mode="self_attn"`:

```text
phase attention over phase vectors, causal mask optional
```

Modify `ResonanceBlock.forward` to return `(x, phases)` when dynamic phase is
enabled.

### Step 3: Implement Phase-Conditioned Q/K FiLM

Modify `ResonanceAttention.forward` signature:

```python
def forward(self, x, resonance, mask=None, phases=None):
```

When `phase_condition_qk="film"`:

```text
q = q * (1 + gamma_q(phases)) + beta_q(phases)
k = k * (1 + gamma_k(phases)) + beta_k(phases)
```

Initialize gamma/beta projections near zero so the model starts close to the
current baseline.

### Step 4: Add Conditions

In `regime_probe.py` condition builder, add:

```text
phase_dynamic_mlp
phase_dynamic_attn
phase_qk_film
phase_dynamic_qk_film
complex_directional
structural_heads_1
```

Each condition should write its exact config into `results.json`.

### Step 5: Smoke Tests

Run tiny tests locally:

```bash
uv run python resonance/structural_task_probe.py \
  --task blimp \
  --data_path data/processed/blimp/wh_island.jsonl \
  --output_dir resonance/outputs/hyper_smoke/wh_island \
  --device cpu \
  --conditions standard,resonance_full_normalized,phase_stream_only_normalized,phase_dynamic_mlp,phase_qk_film \
  --seeds 1 \
  --epochs 1 \
  --train_examples 64 \
  --val_examples 64 \
  --embed_dim 64 \
  --layers 2 \
  --heads 4 \
  --ff_dim 256 \
  --batch_size 16 \
  --max_length 224 \
  --eval_each_epoch
```

Do not start long runs until this passes.

## First Serious Variant Runs

Use `wh_island` because it is the best known regime. Start with n=150.

```bash
python resonance/structural_task_probe.py \
  --task blimp \
  --data_path data/processed/blimp/wh_island.jsonl \
  --output_dir resonance/outputs/hyper/wh_island_n150_variants \
  --device cuda \
  --conditions standard,resonance_full_normalized,phase_stream_only_normalized,resonance_inert_normalized,phase_dynamic_mlp,phase_qk_film,phase_dynamic_qk_film \
  --seeds 11 23 37 41 53 \
  --epochs 5 \
  --train_examples 150 \
  --val_examples 300 \
  --embed_dim 96 \
  --layers 3 \
  --heads 4 \
  --ff_dim 384 \
  --batch_size 16 \
  --max_length 224 \
  --eval_each_epoch
```

Then test n=100:

```bash
python resonance/structural_task_probe.py \
  --task blimp \
  --data_path data/processed/blimp/wh_island.jsonl \
  --output_dir resonance/outputs/hyper/wh_island_n100_variants \
  --device cuda \
  --conditions standard,resonance_full_normalized,phase_stream_only_normalized,resonance_inert_normalized,phase_dynamic_mlp,phase_qk_film,phase_dynamic_qk_film \
  --seeds 11 23 37 41 53 67 79 \
  --epochs 5 \
  --train_examples 100 \
  --val_examples 300 \
  --embed_dim 96 \
  --layers 3 \
  --heads 4 \
  --ff_dim 384 \
  --batch_size 16 \
  --max_length 224 \
  --eval_each_epoch
```

Promote only variants that beat:

- standard,
- current full,
- phase-only,
- inert,
- and eventually iso-param standard.

## Second Serious Runs: Structure-Richer Tasks

The current attention-bias readout may only matter on tasks requiring pairwise
or graph-like matching. Try these after the `wh_island` variant runs:

### Graph Alias

```bash
python resonance/structural_task_probe.py \
  --task graph_alias \
  --output_dir resonance/outputs/hyper/graph_alias_variants \
  --device cuda \
  --conditions standard,resonance_full_normalized,phase_stream_only_normalized,resonance_inert_normalized,phase_dynamic_mlp,phase_qk_film,phase_dynamic_qk_film \
  --seeds 11 23 37 41 53 \
  --epochs 5 \
  --train_examples 1000 \
  --val_examples 500 \
  --embed_dim 96 \
  --layers 3 \
  --heads 4 \
  --ff_dim 384 \
  --batch_size 16 \
  --n_graphs 32 \
  --n_nodes 8 \
  --alias_pool_size 96 \
  --out_degree 2 \
  --walk_length 4 \
  --query_steps 1 \
  --val_graphs new \
  --max_length 256 \
  --eval_each_epoch
```

### COGS / SLOG / CFQ

Use generic probes:

```text
data/processed/probes/cogs/train.jsonl
data/processed/probes/cogs/gen.jsonl
data/processed/probes/slog/slog_cogs_lf_train.jsonl
data/processed/probes/slog/slog_gen_cogs_lf.jsonl
data/processed/probes/cfq/mcd1/train.jsonl
data/processed/probes/cfq/mcd1/test.jsonl
```

Start small. These are more expensive and the current generic yes/no adapter may
be weak.

## Evaluation Gate

A task/variant is interesting only if it satisfies:

- accuracy beats majority by at least 5 points;
- validation loss improves during training;
- hidden-state label geometry has positive `label_gap`;
- it beats inert and phase-only in the way the hypothesis predicts;
- it survives at least 5 seeds.

Interpretation guide:

- **full = phase-only**: the phase stream is doing the useful work; attention
  adapter is not adding value.
- **bias-only = inert**: the relation-bias path is not meaningful in that
  regime.
- **all conditions high**: saturated, not useful.
- **accuracy bump but negative loss/gap**: scouting signal, not a result.
- **new variant beats phase-only and inert**: real architecture candidate.

## Interpretability Checks After A Variant Wins

For a winning variant, build:

1. Phase ablation:
   - zero phase,
   - shuffle phase across tokens,
   - add Gaussian phase noise,
   - freeze semantic stream and alter phase.
2. Phase patching:
   - clean/corrupt pairs;
   - patch phase from clean to corrupt;
   - measure answer-logit recovery.
3. Phase geometry:
   - PCA effective rank;
   - clustering by label/structure;
   - phase drift by layer;
   - relation-matrix effective rank.
4. Quantization:
   - quantize phase path only;
   - quantize semantic path only;
   - compare degradation.
5. Sparse probes/SAE:
   - train SAE on phase states and residual states;
   - compare feature purity and causal faithfulness.

## Short-Term Priorities

1. Finish / pull current overnight runs.
2. Regenerate `resonance/outputs/lab_notebook/lab-notebook.md`.
3. Implement dynamic phase MLP update.
4. Implement phase-conditioned Q/K FiLM.
5. Run `wh_island_n150` and `wh_island_n100` variant comparisons.
6. Add iso-parameter standard baseline.
7. Only then return to larger corpus runs.

## What Not To Do

- Do not spend another day sweeping weak tasks that do not pass the regime gate.
- Do not treat HANS saturation as evidence.
- Do not call the current attention bias the core theory.
- Do not claim quantum/microtubule analogies as evidence. They may inspire
  architecture, but they are not part of the empirical case.
- Do not evaluate only raw accuracy; loss and geometry matter.
- Do not compare a larger phase model to a smaller standard model without an
  iso-parameter control.

## One-Sentence North Star

Build a transformer where the phase stream is an evolving, inspectable
structural state trained against real equivalence/topology objectives, and test
whether that state gives better low-data structural generalization and better
mechanistic access than an ordinary residual stream.
