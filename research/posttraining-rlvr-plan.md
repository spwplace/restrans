# Structural Posttraining / RLVR Plan

Date: 2026-04-30

## Why This Pivot

Raw language-model pretraining is not the clean test of the core hypothesis. A low-dimensional structural stream should matter most when the training signal explicitly rewards structural validity, behavioral equivalence, and process-correct transitions. The current plan therefore treats web-scale LM as a later transfer experiment, not as the main discovery loop.

The open posttraining recipe to crib from is:

1. Supervised warm start on verified examples.
2. Preference-style optimization on correct versus incorrect outputs.
3. Reinforcement learning with verifiable rewards.
4. Evaluation on held-out structural families plus interpretability diagnostics.

This mirrors the practical shape of Tulu-style open posttraining while adapting it to small models and exact formal verifiers.

## Literature Lessons Integrated

- Tulu 3 / open-instruct: use SFT, then DPO, then RL with verifiable rewards rather than relying on pretraining alone.
- RLVR / GRPO: when answers are objectively checkable, replace reward models with verifiers.
- GRPO-VPS / process supervision: sparse outcome reward is weaker than step-level verifiable feedback; this motivates trace-validity tasks.
- LENS / negative groups: failed generations can still be useful if confidence-weighted, so our future free-generation RL should not discard all-wrong groups.
- LamPO / pairwise decomposed advantages: pairwise reward gaps are a natural fit for equivalence classes and contrastive structural groups.
- "GRPO sharpens pretraining biases" style results: RL will not magically create a capability the data distribution never exposed. This is why the task lake needs lambda traces, VM traces, cap matching, protocol closure, code equivalence, and proof/progress data before bigger RL.

## New Verifiable Task Families

Implemented now:

- `lambda_equivalence`: two simply typed lambda terms; label is beta-normal-form equivalence.
- `lambda_beta_step`: current term and proposed successor; label is exact next normal-order beta step.
- `lambda_trace`: full proposed beta-reduction trace; label is exact whole-trace validity. This is the multi-step cascade of the one-step task.
- `vm_step`: current VM stack, instruction, and proposed next stack; label is exact one-instruction transition validity.
- `vm_trace`: stack-machine program and proposed stack trace; label is exact trace validity under a tiny VM interpreter.
- `vm_equivalence`: two stack-machine programs; label is equality of final stack.
- `dfa_equivalence`: two deterministic finite automata; label is exact language equivalence by product-automaton search.

The intended structure is local-to-global: one-step tasks train the transition relation, and trace/equivalence tasks test whether the model can use the closure of that relation. In the current harness this is operationalized as optional SFT curriculum, e.g. `lambda_beta_step -> lambda_trace` and `vm_step -> vm_trace`, before DPO / exact-RL / GRPO-style verifier optimization on the multi-step target.

Existing high-value tasks to keep:

- `cap_matching`: knowledge-based unification / cap membership.
- `algebraic_protocol`: Dolev-Yao-style closure and derivability.
- `unification`: first-order occurs-check unification.

External/common benchmarks to stage later:

- SLOG / COGS / CFQ for compositional semantic parsing.
- HANS / MSGS / BLiMP / Linzen as syntax and shortcut probes.
- EquiBench / CETBench / ContraCode-style transformations for code equivalence.
- LeanProgress / LeanDojo subsets for proof-state progress and theorem dependency structure.
- CLRS for algorithmic trace/procedure learning if we want a standardized algorithm suite.

## Algorithms in the Harness

`resonance/train_structural_posttrain.py` currently supports:

- `sft_only`: answer-token supervised finetuning.
- `dpo`: pairwise correct-vs-wrong answer optimization against a frozen reference.
- `exact_rl`: exact expected verifier reward for the binary answer distribution, with KL and entropy terms.
- `grpo_enum`: enumerated two-answer GRPO-style update with clipped group-relative objective.

The immediate goal is not to crown an RL algorithm. It is to check whether explicit verifier pressure makes the structural stream more useful, more probeable, or more causally relevant than it was under plain pretraining.

## Model Conditions for First Pilots

Keep the first posttraining runs small and pointed:

- `standard_alibi`: strong baseline.
- `phase_dynamic_qk_film_alibi_normalized`: compact phase stream conditions Q/K, with ALiBi addressing restored.
- `relation_value_qk_film_alibi_normalized`: phase stream conditions Q/K and routes values through a relation channel.

Do not spend more time on the old static additive bias as a headline architecture. It remains a historical ablation only.

## Success Criteria

A useful signal requires at least two of:

- higher held-out verifier accuracy/reward than `standard_alibi`;
- better phase-stream label geometry than hidden-state geometry;
- selective phase ablation damage on structural tasks;
- stronger gains on trace/equivalence tasks than on shallow lexical tasks.

Accuracy alone is insufficient. A method that improves score but leaves the phase stream unused is just a performance trick.

## Execution Order

1. Smoke every new task and `train_structural_posttrain.py`.
2. Run a focused local pilot on `lambda_trace`, `vm_trace`, `cap_matching`, and `algebraic_protocol`.
3. If one structural variant beats or matches `standard_alibi` while showing stronger phase geometry or ablation sensitivity, repeat with 3 seeds.
4. Move the same harness to persvati/4090/cloud with larger models and more examples.
5. Add free-generation RL only after the binary verifier regime shows signal.

## Next Engineering Steps

- Add a richer process-reward API so trace tasks can expose per-step correctness, not only final yes/no labels.
- Use the current trace `process_score` metadata for diagnostics and then for generated-trace RL once we move beyond binary answer tokens.
- Add DFA/NFA equivalence and CLRS-like algorithm trace generators.
- Add code-equivalence ingestion for EquiBench/CETBench when acquired.
- Add LeanProgress data adapter when the local Lean toolchain/data are ready.
- Add a grouped equivalence trainer that samples many same-normal-form programs and optimizes pairwise/LamPO-like structural advantages.
