# Small Code Model Standard: Recommendation

Date: 2026-04-30

## Read

There is a credible small/mid-scale code-model track we can replicate, but it is not primarily HumanEval-style generation at first. For our architecture question, the best standard lane is code understanding and semantic equivalence:

- [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) clone detection / code search / defect detection.
- [CodeT5](https://arxiv.org/abs/2109.00859), [CodeT5+](https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/README.md), CodeBERT / [GraphCodeBERT](https://arxiv.org/abs/2009.08366) / [UniXcoder](https://arxiv.org/abs/2203.03850) as external reference baselines.
- EquiBench/CETBench-style equivalence checking as the newer semantic benchmark that better matches our thesis.

HumanEval, MBPP, [MultiPL-E](https://arxiv.org/abs/2208.08227), APPS, [BigCodeBench](https://arxiv.org/abs/2406.15877), [LiveCodeBench](https://arxiv.org/abs/2403.07974), and SWE-bench are real standards for generation, but they are not the right first target for a scratch 20M-100M architecture study. They become useful when we either adapt a pretrained code model or run a 350M-ish decoder with a serious code curriculum.

The practical scale target should now be **125M -> 350M -> 600M**, with **350M as the first serious midpoint**. That lines up with [CodeGen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono), the phi-1-small result in [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644), and the older CodeT5/CodeT5+ size ladder. A 1B-ish model is plausible for inference, adapter training, or a cloud run, but it is not the fastest way to answer the architecture question on nextop + persvati.

## Closest Literature Standard

### CodeXGLUE family

CodeXGLUE is still the main standardized umbrella for code-intelligence tasks. It includes clone detection (BigCloneBench and POJ-104), defect detection (Devign), code search, code completion, code summarization, translation, and refinement. The clone tasks are especially relevant: BigCloneBench is binary clone classification, while POJ-104 is retrieval by problem/equivalence class.

This is the right “small code model standard” because many classic code models report here: CodeBERT, GraphCodeBERT, CodeT5, PLBART, UniXcoder, etc. CodeT5-small is useful as a lower anchor, but the more serious anchors for us are CodeT5-base/CodeT5+ 220M, CodeGen-350M, and CodeT5+/CodeT5-large around 770M.

### Structural-code prior line

GraphCodeBERT injects data-flow information into pretraining and evaluates on code search, clone detection, translation, and refinement. UniXcoder explicitly uses AST-derived structure and contrastive/cross-modal objectives. These are important because they are strong prior art for “code structure helps when represented explicitly.” Our resonance/phase/relation-value stream should be compared against them conceptually, and ideally against their checkpoints as external baselines.

This line also tells us what not to do: do not keep treating the phase stream as a decorative LM add-on. For code, the stream should be pressured by concrete program structure: data-flow edges, AST paths, equivalence classes, verifier traces, and hard semantic negatives.

### Equivalence-specific line

EquiBench is closer to our research thesis than ordinary clone detection: it asks whether programs are semantically equivalent across multiple languages and equivalence categories. This is where the structural stream has the most plausible chance to matter. The benchmark is newer and more LLM-prompting oriented, but we can adapt it into pair classification / retrieval for small encoders.

## What Is In Reach

### In reach now

1. Fine-tune and evaluate 20M-125M scratch models on:
   - POJ-104 problem-classification / retrieval.
   - BigCloneBench binary clone detection.
   - EquiBench pair classification.
   - CodeSearchNet text-code retrieval or comment-code alignment.

2. Add external checkpoint baselines:
   - CodeT5-small (~60M).
   - CodeBERT / GraphCodeBERT (~125M-class).
   - UniXcoder if setup is not too heavy.

3. Run our variants as parameter-matched models:
   - `standard_alibi`.
   - `phase_dynamic_qk_film_alibi_normalized`.
   - `relation_value_qk_film_alibi_normalized`.
   - a graph/AST/data-flow initialized phase variant if we add parser preprocessing.

4. Use metrics that reviewers recognize:
   - BigCloneBench: F1 / accuracy / AUROC.
   - POJ-104: MAP@R / Recall@k / problem-ID accuracy for a cheaper proxy.
   - CodeSearchNet: MRR / Recall@k.
   - EquiBench: accuracy / AUROC by equivalence category.

5. Add a generation sanity check once the model is decoder-capable:
   - HumanEval/MBPP for quick continuity with the literature.
   - MultiPL-E if multilingual/general structural transfer matters.
   - BigCodeBench or LiveCodeBench only after we have a model that can generate nontrivial code; these are too sparse/expensive as the first architecture signal.

### In reach with persvati / nextop plus careful engineering

1. **350M decoder pretraining/posttraining pilot**:
   - Train a GPT/CodeGen-like standard model and the best structural variant on a curated code mixture.
   - Use sequence length 512-1024, BF16/AMP where available, gradient accumulation, checkpointing, and no broad architecture sweep.
   - Target 100M-500M tokens on local hardware as a signal run; use cloud for 1B-5B tokens if the signal is real.

2. **Optional pretrained-backbone adapter experiments**:
   - Load CodeGen-350M or CodeT5+ 220M/770M.
   - Freeze most of the backbone.
   - Add structural adapters / phase-conditioned QK FiLM / relation-value modules.
   - Compare against LoRA/IA3-style adapters at the same trainable parameter count.
   - Fine-tune on BigCloneBench, POJ-104 retrieval, CodeSearchNet retrieval, and EquiBench pair classification.
   - This is **not primary evidence for the architecture hypothesis**. It only asks whether a structural module can be retrofitted onto an already-trained code model.

3. **Verifier/posttraining lane**:
   - Use CodeRL-style feedback, but initially on tasks with cheap verification: generated unit tests, EquiBench equivalence labels, local VM/lambda traces, and property tests.
   - Reward executable correctness or equivalence, not merely next-token likelihood.

### Later / bigger

1. Code generation with HumanEval/MBPP/MultiPL-E.
2. CodeRL-style unit-test reward training.
3. SantaCoder/StarCoderBase/CodeGen-350M scale comparisons.
4. LiveCodeBench/SWE-bench style agentic coding.

These are real standards but not the right first proof of the structural-stream idea. [SantaCoder](https://arxiv.org/abs/2301.03988) is a good example of the 1.1B code-generation lane, and [StarCoder2](https://arxiv.org/abs/2402.19173) shows the modern 3B/7B/15B frontier, but those are comparison points rather than immediate local training targets.

## Recommended Experiment

### Phase 1: establish a credible baseline harness

Goal: reproduce standard small-code-model results enough that the harness is credible.

- Fine-tune CodeT5-small on BigCloneBench and POJ-104 using the official or close-to-official CodeXGLUE setup.
- Fine-tune CodeBERT or GraphCodeBERT on at least BigCloneBench / POJ-104.
- Do not worry if our scratch models are worse; this phase validates data splits and metrics.

### Phase 2: architecture study on matched full models

Goal: ask whether our structural stream helps at fixed parameter/token budget.

- Train 20M, 125M, and then 350M models.
- Pretrain on code tokens from CodeSearchNet / POJ / EquiBench source pools plus permissive GitHub / The Stack-style subsets if acquired.
- Fine-tune on clone/equivalence tasks.
- Compare standard vs phase_dynamic vs relation_value.
- Train all architecture variants end-to-end under the same objective and token budget. This is the primary evidence path for the core hypothesis.

Success is not “beats CodeT5-small.” Success is “beats a matched standard transformer on hard semantic splits and shows phase causality/probeability.”

### Phase 3: structural posttraining

Goal: use verifier-like training pressure, matching the current research direction.

- Build positives from same-problem or equivalent programs.
- Build hard negatives from same lexical/problem context but changed semantics.
- Add contrastive group loss over equivalence classes.
- Add DPO / exact-RL where the reward is an executable/test/equivalence verifier.

Most promising first cells:

- EquiBench OJ_VA with phase_dynamic under exact-RL or DPO.
- POJ-104 retrieval with relation_value pooling.
- BigCloneBench hard negatives with phase permutation diagnostics.

### Phase 4: optional retrofit / transfer study

Goal: ask whether a structural module remains useful when grafted onto an existing code model. This is useful engineering, but it does not prove the main architecture thesis.

- Use CodeGen-350M-mono as the decoder baseline because it is exactly in the desired scale range and has a permissive BSD-3-Clause model card.
- Use CodeT5+ 220M / 770M as encoder-decoder baselines because the official release covers 220M, 770M, 2B, 6B, and 16B models.
- Add the structural stream as an adapter rather than replacing the whole backbone:
  - phase-conditioned Q/K FiLM adapter,
  - relation-value adapter,
  - data-flow/AST-initialized phase adapter.
- Freeze the base for the first pass; then try partial unfreezing.
- Evaluate against LoRA/IA3 adapters with the same trainable parameter budget.

This is a separate follow-up. It should not displace the matched scratch/full-model runs on synthetic, logical, algebraic, and real code-equivalence tasks.

## My Recommendation

Yes, this is in reach, but we should not try to become a tiny HumanEval model yet. The strongest next milestone is:

> “A parameter-matched structural-stream encoder beats a standard transformer on CodeXGLUE/EquiBench semantic clone/equivalence tasks, with phase or relation-value interventions showing causal use of structural information.”

That would matter scientifically even if CodeT5-small remains stronger in absolute terms, because CodeT5-small has far more pretraining and a different objective. The right paper claim would be about architectural/training-regime signal under controlled budgets, not leaderboard SOTA.

With the revised scale target, the better headline becomes:

> “At 125M-350M parameters, structural-stream models or adapters improve controlled code-equivalence and clone-retrieval performance over parameter-matched transformers and standard adapters, with interpretable structural geometry and causal intervention evidence.”

That is much better than another toy-task result and more realistic than training a 1B code model from scratch locally.

## Acquisition Status

Already present in our data lake or manifest:

- POJ-104 via CodeXGLUE HF packaging.
- BigCloneBench via CodeXGLUE HF packaging.
- CodeSearchNet pair data.
- EquiBench.
- COGS/CFQ/BLiMP/BabyLM/TinyStories for non-code transfer.

Need to add or verify:

- Devign defect detection.
- Official CodeXGLUE eval scripts or metric parity.
- CodeT5-small / CodeBERT / GraphCodeBERT checkpoints for external baselines.
- Optional AST/data-flow extraction pipeline for a GraphCodeBERT-style structural baseline.
- CodeGen-350M-mono and CodeT5+ 220M/770M checkpoints for adapter baselines.
- A curated permissive code-pretraining shard if we want to run the 350M scratch lane.

## Hardware Fit

Persvati is idle after reboot and has plenty of disk again. The local repo environment on persvati currently needs its torch/ROCm environment reactivated or rebuilt before training.

Rough feasibility:

- **20M-125M**: local routine work. Good for harness validation and seed variance.
- **350M**: plausible on persvati/nextop for short signal runs if we add BF16/AMP, gradient accumulation, periodic checkpoints, and possibly activation checkpointing. A real 1B-token pretrain will likely take days locally, so use it only after the benchmark harness is credible.
- **600M**: plausible for fine-tuning/adapters; scratch pretraining locally is possible but slow and should move to cloud if it becomes central.
- **1B+**: use pretrained models, quantized inference, adapters, or cloud. Do not make this the first local architecture test.
