# Resonance Transformer Reproduction & Extension Plan

## Overview
Reproduce the resonance transformer architecture from https://github.com/mbilokonsky/resonance, run novel experiments (perturbation stability, compressibility, synthetic data), conduct scale studies, and write an academic paper.

## Stage 1 — Code Reproduction & Baseline
**Goal**: Extract, clean, and reproduce the resonance transformer code in a runnable Python module. Validate on small data.

- Sub-task 1A: Extract clean modules from the notebook (StandardTransformer, ResonanceTransformer, training loop, data loading)
- Sub-task 1B: Run a tiny baseline experiment to verify the architecture works (small data, few epochs)
- Sub-task 1C: Validate that phonetic init and resonance attention both function

## Stage 2 — Novel Experiments Implementation
**Goal**: Implement the user's specific research ideas.

- Sub-task 2A: **Perplexity Stability Under Perturbation**
  - Perturb phase embeddings, semantic embeddings, and full weights with Gaussian noise at various scales
  - Measure perplexity delta vs perturbation magnitude
  - Compare Standard vs Resonance models

- Sub-task 2B: **Compressibility of Learned Parameters**
  - Quantization: INT8, INT4 weight quantization
  - Pruning: Magnitude-based and structured pruning of phase/semantic components
  - Measure perplexity vs compression ratio
  - SVD analysis of embedding matrices

- Sub-task 2C: **Compressibility of "Gestalt Streams"**
  - Extract phase-space representations (resonance matrices)
  - Measure mutual information between phase and semantic streams
  - Test if phase stream can be compressed independently
  - Analyze dimensionality of phase subspace via PCA

- Sub-task 2D: **Synthetic Data Generator (Lambda Calculus)**
  - Generate well-typed lambda terms
  - Generate proof walk datasets (statement → proof path → program synthesis)
  - Implement contrastive group training as described
  - Implement semantic-preserving mutation (bisimilar programs)

## Stage 3 — Scale Study
**Goal**: Run systematic scaling experiments.

- Sub-task 3A: Define scale grid (1x, 2x, 5x, 10x, 20x, 50x, 100x) in terms of model size and/or data
- Sub-task 3B: Train at each scale, measure scaling laws (loss vs params, loss vs compute)
- Sub-task 3C: Compare Standard vs Resonance scaling curves

## Stage 4 — Paper Writing
**Goal**: Write a rigorous academic paper.

- Sub-task 4A: Read paper-writing skill, design outline
- Sub-task 4B: Write sections: Abstract, Introduction, Related Work, Methods, Experiments, Results, Discussion, Conclusion
- Sub-task 4C: Compile to DOCX

## File Propagation
- Stage 1 → Stage 2: `resonance/` package, baseline model checkpoints
- Stage 2 → Stage 3: Experiment harness, synthetic data generators
- Stage 3 → Stage 4: All experiment results, figures, tables
- Final: Paper `.md` and `.docx`

## Working Directory
`/mnt/agents/output/resonance/` for code, data, models, results.
