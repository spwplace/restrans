# Sequence Verifier Status - 2026-05-01

## Why We Pivoted

The completed cap-matching expansion is useful but not clean enough to carry
the paper: the older generator has shortcut-prone negatives, and the strongest
condition across the sweep is often the DeBERTa-lite relative-attention
baseline. We therefore moved the active workload toward exact sequence
transformer tasks in the original formal-methods neighborhood:

- lambda one-step beta reduction,
- lambda multi-step trace validation,
- stack-VM one-step transition validation,
- stack-VM multi-step trace validation,
- lambda/VM behavioral equivalence,
- exact-RL posttraining with one-step curriculum into trace validation,
- target-sequence prediction for next formal state / normal form.

## Completed Cap Sweep

Summary file:

- `lab_notebook_hyper_live/cap_matching_expansion_summary.md`

High-level read:

- Cap matching is learnable and non-saturated.
- `standard_deberta_lite` wins depth4, depth4->6, and depth6.
- `phase_dynamic_qk_film_alibi_normalized` wins depth5 and depth5->7.
- `harmonic_relation_value_normalized` wins depth7.
- Treat as a signal probe, not a headline result.

## New Code Added

- `resonance/train_sequence_prediction.py`
  - Exact target-sequence training for `lambda_beta_next`,
    `lambda_normal_form`, and `vm_step_next`.
  - Uses causal alignment: logits before a target token predict that token.
  - Reports target-token accuracy and exact-match accuracy.

- `scripts/run_sequence_verifier_nextop.sh`
  - Nextop-sized verifier classification and exact-RL suite.

- `scripts/run_sequence_verifier_persvati.sh`
  - Larger persvati ROCm verifier classification and exact-RL suite.

- `scripts/run_sequence_prediction_nextop.sh`
  - Nextop-sized target-sequence prediction suite.

- `scripts/run_sequence_prediction_persvati.sh`
  - Larger persvati target-sequence prediction suite.

## Smoke Tests

Passed:

- `resonance/train_structural_posttrain.py` on `vm_trace` with `vm_step`
  curriculum, exact-RL, cascade eval, and phase ablation.
- `resonance/structural_task_probe.py` on `lambda_trace`.
- `resonance/train_sequence_prediction.py` on `vm_step_next`.
- `resonance/train_sequence_prediction.py` on `lambda_beta_next`.

One draft sequence-prediction bug was found and fixed: target tokens were
initially scored at their own teacher-forced positions. The trainer now scores
each target token from the previous position.

## Active Runs

Persvati:

- tmux session: `seq_verify`
- output root:
  `resonance/outputs/sequence_verifier_persvati_2026_05_01_seqverify`
- current script: `scripts/run_sequence_verifier_persvati.sh`
- first task at launch: `lambda_beta_step_depth6`

Persvati queue:

- tmux session: `seq_predict_queue`
- waits for `seq_verify` to finish, then runs:
  `scripts/run_sequence_prediction_persvati.sh`
- future output root:
  `resonance/outputs/sequence_prediction_persvati_2026_05_01_seqpredict`

Nextop:

- tmux session: `seq_verify_nextop`
- output root:
  `resonance/outputs/sequence_verifier_2026_05_01_092852`
- current script: `scripts/run_sequence_verifier_nextop.sh`

Nextop queue:

- tmux session: `seq_predict_nextop_queue`
- waits for `seq_verify_nextop` to finish, then runs:
  `scripts/run_sequence_prediction_nextop.sh`
- future output root:
  `resonance/outputs/sequence_prediction_2026_05_01_seqpredict_nextop`

## Interpretation Contract

The next credible positive result is not "phase beats baseline on one toy
classifier." It is one of:

- phase/structural variants improve trace or target-sequence extrapolation
  compared with ALiBi and DeBERTa-lite;
- exact-RL or one-step curriculum disproportionately benefits structural
  variants on multi-step trace/cascade tasks;
- phase ablation hurts structure-sensitive decisions selectively;
- sequence prediction shows lower exact-match loss or better extrapolation for
  a structural variant after enough training.

If DeBERTa-lite or ALiBi remain stronger across these tasks, the honest result
is that our current structural-stream parameterization is not yet better than
known relation-aware attention baselines.
