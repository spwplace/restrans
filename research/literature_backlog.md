# Literature Backlog And Ownership Tracker

Date: 2026-04-28

This file tracks every research direction surfaced by the literature report and
where it enters the codebase. Nothing here is a novelty claim; it is an
execution checklist.

## Architecture And Baselines

| Direction | Why it matters | Status | Files |
|---|---|---|---|
| Vanilla transformer baseline | Basic learnability and loss control. | Implemented. | `resonance/resonance/models.py` |
| Iso-parameter baseline | Controls phase-stream parameter count. | Implemented in structural probes. | `resonance/story_topology_eval.py` |
| ALiBi / relative-bias baseline | Controls for distance/addressing priors. | Implemented as `standard_alibi`. | `resonance/resonance/models.py`, `resonance/story_topology_eval.py` |
| DeBERTa-style split baseline | Controls against content/position disentanglement. | Implemented as `standard_deberta_lite`; intentionally a compact comparator, not full DeBERTa. | `resonance/resonance/models.py` |
| Abstractor / DAT-style relational stream | Closest prior-art neighborhood for explicit relational pathways. | First comparator implemented as `relational_stream_lite`; stronger parity implementation still pending. | `resonance/story_topology_eval.py` |
| Graphormer / GRPE graph bias | Required when ground-truth graph structure is known. | Pending. Needs graph-specific runner support. | planned |
| Structural stream without direct attention path | Separates representation from causal attention bias. | Implemented as `phase_stream_only*`. | model/runners |
| Legacy `R` logit bias | Tests original scalar additive mechanism. | Implemented as `resonance_full*` / `bias_only*`. | model/runners |
| Dynamic structural state | Tests phase as context-evolving scratchpad. | Implemented as `phase_dynamic_mlp`, `phase_dynamic_attn`. | model/runners |
| Phase-conditioned Q/K | Tests phase in attention geometry rather than as external logit bonus. | Implemented as `phase_qk_film`, `phase_dynamic_qk_film`. | model/runners |
| Directional relation kernel | Tests asymmetric dependencies: binder/use, cause/effect, proof step flow. | Implemented as `complex_directional*`. | `resonance/resonance/kernels.py` |
| Dedicated structural heads | Tests architectural disentanglement across heads. | Implemented as `structural_heads_1*`. | model/runners |
| Sparse/bottlenecked structural stream | Integrates intrinsic-interpretability lessons. | Pending as training regularizer. SAE example exists. | planned |
| Pair-specific learned relation features | Stronger relation-aware comparator. | Pending; likely expensive and should follow medium results. | planned |

## Training Pressure

| Direction | Why it matters | Status | Files |
|---|---|---|---|
| Architecture-only training | Tests whether the stream self-organizes under task loss. | Implemented. | all runners |
| Supervised structural task loss | Most structural probes are already supervised via answer label. | Implemented at task level. | `resonance/regime_probe.py` |
| Phase contrastive objective | Tests whether explicit structural pressure is required. | Implemented by condition suffix `*_phase_contrastive` or `--phase_contrastive_weight`. | `resonance/regime_probe.py`, `resonance/structural_task_probe.py` |
| Same-graph/story contrastive objective | Natural-language bridge objective. | Implemented in story topology runner. | `resonance/story_topology_eval.py` |
| Semantic-preserving program contrast | Core code-equivalence objective. | EquiBench downloader/converter implemented; CETBench still pending location confirmation. | `resonance/download_datasets.py`, `resonance/prepare_external_probe_data.py` |
| Parser/dependency supervision | Syntax-aware structural target. | Pending; needs parser outputs or dependency datasets wired into probes. | planned |
| Graph/Laplacian phase initialization | Tests known structural priors vs learned phase. | Pending. | planned |
| Proof-state / proof-progress supervision | Formal reasoning pressure. | Pending dataset staging. | LeanProgress/LeanDojo planned |

## Benchmark Families

| Family | Role | Status |
|---|---|---|
| Dyck / cross-Dyck | Cheap stack/matching canary. | Implemented. |
| ListOps / SCAN | Cheap parse/composition canaries. | ListOps-style generator implemented; SCAN pending. |
| Linzen agreement | Canonical syntax probe. | Staged. |
| BLiMP | Minimal-pair structural syntax/semantics. | Staged and active. |
| MSGS | Surface-vs-linguistic shortcut diagnostic. | Staged and active. |
| HANS | NLI heuristic-control diagnostic. | Staged and active. |
| COGS | Semantic parsing compositional generalization. | Staged as generic probe. |
| SLOG | Structural generalization over semantic parsing. | Staged as generic probe. |
| CFQ | MCD compositional semantic parsing. | Staged as generic probe. |
| BabyLM / TinyStories | Natural language transfer corpora. | Staged. |
| AMR / MASSIVE-AMR | Real graph semantics. | Public derivative staged; LDC AMR blocked. |
| Cap matching | Knowledge-based unification from local formal-methods line. | Implemented. |
| First-order unification | Algebraic equivalence under substitution. | Implemented. |
| Algebraic protocol closure | Symbolic reachability / constructor-destructor closure. | Implemented. |
| Graph aliases | Latent graph under surface aliasing. | Implemented. |
| Structural paraphrase / causal stories | Natural-language bridge with hard negatives. | Prototype implemented; stronger verifier needed. |
| POJ-104 / BigCloneBench | Code-class and clone/proxy structure. | Staged where available. |
| EquiBench / CETBench | Better program-equivalence standard. | EquiBench acquisition/conversion implemented; CETBench pending. |
| LeanProgress / LeanDojo | Proof-state and proof-progress structure. | LeanDojo-v2 source/setup script implemented; full runner still pending after sample export. |
| OGB / CLRS | Graph/algorithmic reasoning alternatives. | Pending; lower priority for first paper. |

## Interpretability And Geometry

| Direction | Question | Status |
|---|---|---|
| Residual-vs-phase probes | Is structure more linearly accessible in the structural stream? | Basic hidden geometry implemented; stream-specific probes need expansion. |
| Pairwise relation probes | Does `R` predict real arcs/relations after controlling for distance? | Pending. |
| Effective rank / eigenspectrum | Is the stream compact on structure-rich data? | Basic resonance rank implemented; layerwise stream ranks pending. |
| CKA / SVCCA | Is phase redundant with residual states? | Pending. |
| Phase ablation / shuffle / noise | Is the stream causally used? | Implemented for story runners; needs generic structural runner integration. |
| Activation patching | Do structural decisions move with phase states? | Row-patching primitives implemented; full paired-example patcher pending. |
| SAE on phase vs residual | Is decomposition cleaner in the structural stream? | Example exists; production runner pending. |
| Sparse vs dense geometry | Avoid assuming all features are sparse axes. | Pending archetype/convex-region probes. |
| Quantization/compression | Does the structural stream compress differently? | Earlier experiment code exists; needs integration with new medium suite. |
| Phase steering / ReFT-like edits | Can structural behavior be steered selectively? | Pending after a stable stream is found. |

### Current Interpretability Software

Medium-suite checkpoints can now be saved with `--save_models` or
`SAVE_MODELS=1`.  The handoff audit runner is:

```bash
bash scripts/run_interpretability_on_suite.sh <suite-output-dir>
```

It writes `interpretability_audit/results.json` and `report.md` with:

- original validation accuracy/loss,
- phase zero / permute / noise sensitivity when applicable,
- resonance-weight scale sensitivity when applicable,
- residual-vs-phase answer-state effective ranks and label geometry,
- one-batch attention/resonance trace diagnostics.

This is the minimum causal/geometry bundle for cloud sweeps.  Full SAE,
CKA/SVCCA, and activation patching remain follow-up analysis once a stable
task-family/variant pair is identified.

## Current Medium-Suite Policy

Use `scripts/run_literature_medium_suite.sh` for the first literature-grounded
medium matrix. It deliberately includes:

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

Proceed to larger cloud runs only when a task family shows selective gains,
stream-specific geometry, and causal ablation sensitivity.
