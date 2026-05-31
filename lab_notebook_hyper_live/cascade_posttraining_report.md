# Cascade Posttraining Status

Date: 2026-04-30

## Run Inventory

- nextop: `posttraining_cascade_nextop_2026_04_30_144727`, DPO, MPS, complete.
- persvati: `posttraining_cascade_persvati_2026_04_30_144727`, exact-RL and GRPO-enum, ROCm, complete.
- Results synced locally; persvati HF cache cleanup restored disk from 99% used to 33% used.

## Core Read

The posttraining pivot produced meaningful signal, but not a universal architecture win. `standard_alibi` remains the strongest direct answer classifier on many tasks, especially lambda traces and DPO cap matching. The useful new evidence is narrower: explicit verifier training makes the phase stream measurable and sometimes causal, and relation-value routing sometimes composes one-step judgments into better trace decisions even when direct answer accuracy is not best.

Most important signals:

- `phase_dynamic` has consistently positive phase label geometry on all structural tasks where the phase stream exists.
- Phase permutation hurts several runs, especially persvati exact-RL DFA equivalence (`-0.263`), persvati GRPO algebraic protocol (`-0.145`), nextop DPO algebraic protocol (`-0.055`), and exact-RL cap matching (`-0.086`). This is actual causal usage, not just a pretty probe.
- Direct trace answer accuracy still favors `standard_alibi` on lambda traces. The architecture has not yet beaten the baseline there.
- Cascade evaluation is more interesting: nextop relation-value routing improves composed local-step trace accuracy on lambda (`0.633` vs standard `0.553`) and VM (`0.731` vs standard `0.572`). Persvati replicates a VM cascade advantage for exact-RL relation-value (`0.646` vs standard `0.500`), but not lambda.
- VM trace remains close to chance as a direct yes/no task. That is a useful hard target for dense process rewards and generated-trace RL.

## Accuracy / Geometry Table

| Machine | Alg | Task | Condition | Acc | Gain | Cascade | Phase Gap | Hidden Gap | Permute Δ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| nextop | dpo | algebraic_protocol | phase_dynamic | 0.848 | 0.400 |  | 0.818 | 1.080 | -0.055 |
| nextop | dpo | algebraic_protocol | relation_value | 0.853 | 0.312 |  | -0.033 | 1.200 | -0.141 |
| nextop | dpo | algebraic_protocol | standard | 0.881 | 0.398 |  |  | 1.323 |  |
| nextop | dpo | cap_matching | phase_dynamic | 0.570 | 0.033 |  | 0.890 | -0.288 | 0.019 |
| nextop | dpo | cap_matching | relation_value | 0.727 | 0.198 |  | -0.032 | 0.210 | -0.195 |
| nextop | dpo | cap_matching | standard | 0.748 | 0.242 |  |  | 0.417 |  |
| nextop | dpo | dfa_equivalence | phase_dynamic | 0.759 | 0.138 |  | 0.951 | 0.205 | 0.000 |
| nextop | dpo | dfa_equivalence | relation_value | 0.759 | 0.402 |  | -0.019 | 0.217 | 0.000 |
| nextop | dpo | dfa_equivalence | standard | 0.759 | 0.259 |  |  | 0.188 |  |
| nextop | dpo | lambda_trace | phase_dynamic | 0.811 | 0.328 | 0.573 | 0.767 | 0.734 | -0.084 |
| nextop | dpo | lambda_trace | relation_value | 0.828 | 0.291 | 0.633 | -0.037 | 0.839 | -0.077 |
| nextop | dpo | lambda_trace | standard | 0.842 | 0.344 | 0.553 |  | 1.011 |  |
| nextop | dpo | vm_trace | phase_dynamic | 0.544 | 0.020 | 0.688 | 0.918 | -0.096 | -0.014 |
| nextop | dpo | vm_trace | relation_value | 0.547 | 0.044 | 0.731 | -0.045 | -0.169 | -0.022 |
| nextop | dpo | vm_trace | standard | 0.553 | 0.055 | 0.572 |  | -0.513 |  |
| persvati | exact_rl | algebraic_protocol | phase_dynamic | 0.801 | 0.306 |  | 1.169 | 1.556 | -0.078 |
| persvati | exact_rl | algebraic_protocol | relation_value | 0.844 | 0.323 |  | -0.022 | 1.670 | -0.145 |
| persvati | exact_rl | algebraic_protocol | standard | 0.848 | 0.405 |  |  | 1.687 |  |
| persvati | exact_rl | cap_matching | phase_dynamic | 0.714 | 0.186 |  | 1.055 | -0.328 | -0.086 |
| persvati | exact_rl | cap_matching | relation_value | 0.538 | 0.016 |  | -0.017 | -0.005 | 0.014 |
| persvati | exact_rl | cap_matching | standard | 0.542 | 0.085 |  |  | -0.008 |  |
| persvati | exact_rl | dfa_equivalence | phase_dynamic | 0.763 | 0.257 |  | 1.011 | 1.412 | -0.263 |
| persvati | exact_rl | dfa_equivalence | relation_value | 0.763 | 0.225 |  | -0.020 | 1.389 | -0.001 |
| persvati | exact_rl | dfa_equivalence | standard | 0.763 | 0.322 |  |  | 1.172 |  |
| persvati | exact_rl | lambda_trace | phase_dynamic | 0.794 | 0.315 | 0.505 | 0.531 | 1.318 | -0.009 |
| persvati | exact_rl | lambda_trace | relation_value | 0.816 | 0.348 | 0.505 | -0.033 | 1.610 | -0.083 |
| persvati | exact_rl | lambda_trace | standard | 0.826 | 0.257 | 0.509 |  | 1.643 |  |
| persvati | exact_rl | vm_trace | phase_dynamic | 0.569 | 0.086 | 0.503 | 0.821 | -0.406 | -0.062 |
| persvati | exact_rl | vm_trace | relation_value | 0.508 | 0.013 | 0.646 | -0.041 | -0.274 | 0.012 |
| persvati | exact_rl | vm_trace | standard | 0.535 | 0.035 | 0.500 |  | -0.633 |  |
| persvati | grpo_enum | algebraic_protocol | phase_dynamic | 0.858 | 0.363 |  | 1.186 | 0.865 | -0.145 |
| persvati | grpo_enum | algebraic_protocol | relation_value | 0.849 | 0.328 |  | -0.023 | 1.146 | -0.074 |
| persvati | grpo_enum | algebraic_protocol | standard | 0.829 | 0.387 |  |  | 1.176 |  |
| persvati | grpo_enum | cap_matching | phase_dynamic | 0.656 | 0.129 |  | 1.080 | -0.017 | -0.091 |
| persvati | grpo_enum | cap_matching | relation_value | 0.727 | 0.204 |  | -0.015 | 0.078 | -0.072 |
| persvati | grpo_enum | cap_matching | standard | 0.758 | 0.301 |  |  | 0.137 |  |
| persvati | grpo_enum | dfa_equivalence | phase_dynamic | 0.763 | 0.257 |  | 1.003 | 0.230 | -0.004 |
| persvati | grpo_enum | dfa_equivalence | relation_value | 0.763 | 0.225 |  | -0.020 | 0.240 | 0.000 |
| persvati | grpo_enum | dfa_equivalence | standard | 0.763 | 0.322 |  |  | 0.260 |  |
| persvati | grpo_enum | lambda_trace | phase_dynamic | 0.798 | 0.319 | 0.508 | 0.532 | 0.472 | -0.020 |
| persvati | grpo_enum | lambda_trace | relation_value | 0.797 | 0.328 | 0.504 | -0.033 | 0.925 | -0.085 |
| persvati | grpo_enum | lambda_trace | standard | 0.848 | 0.279 | 0.508 |  | 0.726 |  |
| persvati | grpo_enum | vm_trace | phase_dynamic | 0.531 | 0.048 | 0.507 | 0.841 | -0.028 | -0.018 |
| persvati | grpo_enum | vm_trace | relation_value | 0.531 | 0.036 | 0.520 | -0.039 | -0.037 | 0.000 |
| persvati | grpo_enum | vm_trace | standard | 0.553 | 0.053 | 0.500 |  | -0.045 |  |

## Best Direct Accuracy By Task

| Machine | Alg | Task | Best Condition | Acc | Runner-Up | Acc |
|---|---|---|---|---:|---|---:|
| nextop | dpo | algebraic_protocol | standard | 0.881 | relation_value | 0.853 |
| nextop | dpo | cap_matching | standard | 0.748 | relation_value | 0.727 |
| nextop | dpo | dfa_equivalence | standard | 0.759 | phase_dynamic | 0.759 |
| nextop | dpo | lambda_trace | standard | 0.842 | relation_value | 0.828 |
| nextop | dpo | vm_trace | standard | 0.553 | relation_value | 0.547 |
| persvati | exact_rl | algebraic_protocol | standard | 0.848 | relation_value | 0.844 |
| persvati | exact_rl | cap_matching | phase_dynamic | 0.714 | standard | 0.542 |
| persvati | exact_rl | dfa_equivalence | standard | 0.763 | phase_dynamic | 0.763 |
| persvati | exact_rl | lambda_trace | standard | 0.826 | relation_value | 0.816 |
| persvati | exact_rl | vm_trace | phase_dynamic | 0.569 | standard | 0.535 |
| persvati | grpo_enum | algebraic_protocol | phase_dynamic | 0.858 | relation_value | 0.849 |
| persvati | grpo_enum | cap_matching | standard | 0.758 | relation_value | 0.727 |
| persvati | grpo_enum | dfa_equivalence | standard | 0.763 | phase_dynamic | 0.763 |
| persvati | grpo_enum | lambda_trace | standard | 0.848 | phase_dynamic | 0.798 |
| persvati | grpo_enum | vm_trace | standard | 0.553 | phase_dynamic | 0.531 |

## Interpretation

The strongest thing to pursue is not the old static resonance bias. The strongest candidate mechanisms are:

1. `phase_dynamic` under exact verifier pressure, especially where phase permutation damages accuracy and phase geometry is clean.
2. `relation_value` as a compositional local-step aggregator, judged by `cascade_acc`, not just direct yes/no accuracy.
3. Dense process reward: the trace tasks now expose `process_score`, but the current trainer mostly uses final binary labels. The next version should train directly on local step correctness and terminal validity.

The weak spots are also clear:

- DFA equivalence has identical final accuracy for all conditions in several settings; treat it as suspicious until we inspect label artifacts and decision modes.
- VM trace direct classification is too hard or the binary presentation is too lossy. It should become a generated-trace/process-reward task.
- Lambda trace direct accuracy is learnable, but the standard baseline is still ahead; the cascade diagnostic says local composition may be a different capability than the direct classifier.

## Recommended Next Experiments

1. Add a dense process-loss stage: supervise every local transition in traces, not only the final yes/no answer.
2. Split trace tasks into two heads/evals: direct whole-trace classifier vs composed one-step verifier. The second is closer to the thesis.
3. Run 3-seed focused confirmations only on promising cells: exact-RL `phase_dynamic` cap matching and VM trace; GRPO `phase_dynamic` algebraic protocol; DPO/Exact `relation_value` cascade traces.
4. Make VM trace easier-to-hard curriculum by length (`3 -> 5 -> 7 -> 9`) and report length extrapolation.
5. Inspect DFA generation because equal accuracies suggest a shortcut or class-prior behavior.