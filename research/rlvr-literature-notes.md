# RLVR / Posttraining Notes For This Project

Date: 2026-04-30

These notes translate recent open posttraining work into concrete choices for the resonance/structural-stream project. The purpose is not to reproduce frontier RLHF infrastructure; it is to adopt the parts that match our exact-verifier setting.

## Sources Checked

- Tulu 3 technical post: https://allenai.org/blog/tulu-3-technical
- Tulu 3 paper: https://arxiv.org/abs/2411.15124
- GRPO-VPS: https://openreview.net/forum?id=Ise1xvtUF6
- Don't Waste Mistakes / LENS: https://openreview.net/forum?id=RppsCECZvP
- LamPO: https://openreview.net/forum?id=53btjz7Rf7
- Can GRPO Help LLMs Transcend Their Pretraining Origin?: https://openreview.net/forum?id=9fwvcl0Jur
- Revisiting GRPO on/off-policy: https://openreview.net/forum?id=XXfOf22o3K
- Effective Reinforcement Learning for Reasoning in Language Models: https://icml.cc/virtual/2025/52620

## Takeaways

1. **Use SFT before RL.** Tulu-style recipes do not jump straight from base model to RLVR. They warm-start with supervised data and often prefer a DPO checkpoint before RLVR.

2. **Use exact verifiers whenever possible.** Our formal tasks are unusually well suited to RLVR: beta reduction, VM execution, DFA equivalence, cap matching, unification, and protocol closure all provide deterministic rewards.

3. **Process reward matters.** GRPO-VPS argues for segment/process-level verifiable supervision instead of only sparse final correctness. This directly motivates `lambda_trace` and `vm_trace`, and the next step is exposing per-step correctness as dense reward.

4. **Negative samples should not be wasted.** LENS argues that all-wrong groups can still carry signal through confidence-weighted penalties. For our current binary-answer tasks this is less urgent, but it becomes important when we move to free-form generated traces or programs.

5. **Pairwise / relational advantages fit equivalence classes.** LamPO-style pairwise decomposed advantages are a strong conceptual match for groups of programs that share a normal form or final behavior.

6. **RL sharpens what the model can already represent.** The GRPO generalization paper warns that RLVR often amplifies pretraining biases rather than discovering wholly new capabilities. This supports our data strategy: build rich synthetic and benchmark distributions before scaling RL.

7. **Do not over-index on one algorithm.** Controlled comparisons and implementation studies keep finding ranking inversions across DPO/PPO/GRPO-style methods. Our harness therefore implements DPO, exact expected reward, and enumerated GRPO under one infrastructure.

## Concrete Changes Made

- Added `resonance/train_structural_posttrain.py`.
- Added `lambda_equivalence`, `lambda_beta_step`, `lambda_trace`.
- Added `vm_trace`, `vm_equivalence`.
- Added `dfa_equivalence`.
- Added `scripts/run_structural_posttraining.sh`.
- Added `resonance/summarize_posttraining.py`.

## Next Algorithmic Additions

- Dense process reward for traces: reward each valid prefix transition, not only the final answer.
- Grouped equivalence RL: sample K variants from one behavior class and K negatives, then optimize pairwise advantages.
- Rejection sampling / best-of-N trace generation: generate several traces, verify them, and train on accepted outputs.
- Confidence-weighted negative updates for free-form generations, following the LENS lesson.
- Off-policy replay for expensive generated traces, following the off-policy GRPO motivation.
