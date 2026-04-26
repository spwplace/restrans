# Foundational Experiment Suite: Phase as Equivalence Carrier

> **Date:** 2026-04-26  
> **Platform:** Apple M2 Max, 96GB RAM, MPS  
> **Duration:** ~4–6 hours for full battery (parallelizable)  

## The Core Question

Can a compact learned phase code represent semantic-preserving equivalence structure more efficiently or more robustly than ordinary learned embeddings, relation biases, or explicit graph encodings?

This suite breaks that question into six falsifiable experiments. Each experiment is designed to run on a laptop in minutes to hours, uses proper controls, and produces a single quantitative answer.

---

## Experiment 1: Equivalence Retrieval Benchmark

**Question:** Does the ResonanceTransformer retrieve semantically equivalent programs better than a standard transformer or a dual-head encoder?

**Method:**
1. Generate proof-walk groups with **normal-form verification** and **deduplication**.
2. Train four conditions (same seed, same data, iso-parameter where possible):
   - `standard` — StandardTransformer encoder
   - `dual_head` — Separate semantic + phase embeddings, no attention bias
   - `resonance_full` — Full ResonanceTransformer
   - `inert` — ResonanceTransformer with both mechanisms disabled
3. Evaluate on held-out groups: for each program, find its k nearest neighbors in embedding space. Compute recall@k for same-normal-form matches.

**Expected:** If the phase mechanism helps equivalence retrieval, `resonance_full` > `dual_head` > `standard` > `inert`.

**Control:** Same parameter budget (adjust embed_dim to match). Same tokenizer. Same contrastive objective.

---

## Experiment 2: Phase Geometry & Compressibility

**Question:** Do phase embeddings learn a lower-dimensional, semantically structured representation compared to semantic embeddings?

**Method:**
1. Train a single ResonanceTransformer.
2. Extract semantic and phase embedding matrices.
3. Compute for each:
   - Effective rank (SVD, 95% variance threshold)
   - Clustering quality (silhouette score by POS / phonetic rhyme / word length)
   - Cross-predictability (Ridge regression: predict semantic from phase, phase from semantic)
4. Compare to a random initialization baseline.

**Expected:** Phase has lower effective rank but comparable or better cross-predictability. Semantic has higher rank but better standalone predictive power.

---

## Experiment 3: Perturbation Intervention Battery

**Question:** Is the phase subsystem behaviorally used, or are the extra parameters inert?

**Method:**
1. Train StandardTransformer and ResonanceTransformer to convergence.
2. At inference, apply controlled perturbations:
   - Zero out phase embeddings
   - Permute phase embeddings (destroy learned structure, preserve marginal statistics)
   - Add Gaussian noise to phase (σ = 0.05, 0.20)
   - Same three operations on semantic embeddings (control)
3. Measure PPL delta on validation set.

**Expected:** If phase matters, phase-zero and phase-permute should cause large PPL degradation on ResonanceTransformer but not StandardTransformer. Semantic perturbations should degrade both similarly.

---

## Experiment 4: Iso-Parameter Architecture Search

**Question:** What is the optimal split of parameters between semantic and phase dimensions at a fixed budget?

**Method:**
1. Fix total parameters at ~500K.
2. Grid search over:
   - embed_dim: 128, 192, 256
   - n_frequencies: 8, 16, 32, 64
   - n_layers: 2, 4
3. Adjust ff_dim to keep total params constant.
4. Train each configuration for 10 epochs on synthetic LM.

**Expected:** An intermediate split (not 100% semantic, not 100% phase) achieves lowest PPL.

---

## Experiment 5: Attention Bias Function Analysis

**Question:** Does the resonance bias capture pairwise relationships that QK^T misses?

**Method:**
1. Load a trained ResonanceTransformer.
2. Sample 100 validation sequences.
3. For each sequence, extract:
   - `QK[i,j]` = attention logits before bias
   - `R[i,j]` = resonance matrix
   - `combined[i,j]` = QK + R·w
4. Compute Pearson correlation between |QK| and |R| per head.
5. Identify token pairs where |R| >> |QK| (resonance "fills in").

**Expected:** Correlation < 0.5, indicating orthogonal information. High-R/low-QK pairs should be semantically related (e.g., antonyms, coreference, rhyme).

---

## Experiment 6: Multi-Seed Statistical Validation

**Question:** Is any observed resonance advantage reproducible across random seeds?

**Method:**
1. Run Experiments 1 or a simplified LM task with 5 seeds (11, 23, 42, 57, 89).
2. Report: mean ± std, 95% CI, Cohen's d effect size.
3. Use paired t-test (same seed, different architecture).

**Threshold for claiming an effect:** Cohen's d > 0.5 (medium) and p < 0.05.

---

## Implementation Plan

### Files to create:
- `resonance/foundational_suite.py` — Main experiment runner
- `resonance/foundational/iso_param.py` — Iso-parameter model builder
- `resonance/foundational/geometry.py` — Phase geometry analysis
- `resonance/foundational/perturbation.py` — Perturbation battery
- `resonance/foundational/retrieval.py` — Equivalence retrieval benchmark
- `resonance/foundational/bias_analysis.py` — Attention bias function analysis
- `resonance/foundational/stats.py` — Statistical reporting utilities

### Files to modify:
- `synthetic/proof_walk_generator.py` — Add normal-form deduplication option
- `research_eval.py` — Save checkpoints for intervention reuse

---

## Running Order (M2)

```bash
# 1. Phase geometry (fast, analytical, uses existing checkpoint if available)
python -m foundational_suite --experiment geometry --checkpoint path/to/checkpoint.pt

# 2. Multi-seed ablation (moderate, ~30 min per seed × 5 seeds = 2.5 hrs)
python -m foundational_suite --experiment ablation --seeds 11 23 42 57 89 --epochs 10

# 3. Perturbation battery (fast, uses models from step 2)
python -m foundational_suite --experiment perturbation --checkpoint_dir outputs/ablation/

# 4. Equivalence retrieval (moderate, ~1 hr)
python -m foundational_suite --experiment retrieval --epochs 10 --seeds 11 23 42

# 5. Attention bias analysis (fast, analytical)
python -m foundational_suite --experiment bias_analysis --checkpoint outputs/ablation/resonance_full/seed42/latest.pt

# 6. Iso-parameter search (longer, ~3 hrs for full grid)
python -m foundational_suite --experiment iso_param --epochs 10

# Or run everything:
python -m foundational_suite --all
```

---

## Success Criteria

| Experiment | Pass Threshold | What It Would Mean |
|---|---|---|
| 1. Retrieval | Resonance recall@5 > Standard by >5% | Phase helps equivalence retrieval |
| 2. Geometry | Phase rank < 0.5 × semantic rank | Phase learns compressible structure |
| 3. Perturbation | Phase-permute PPL delta > 2× semantic-permute delta | Phase is behaviorally used |
| 4. Iso-param | Intermediate split beats extremes | There is an optimal phase budget |
| 5. Bias analysis | Correlation(QK, R) < 0.5 | R captures orthogonal information |
| 6. Statistics | Cohen's d > 0.5, p < 0.05 | Effect is reproducible |

If **≥4 of 6** pass, we have publishable evidence for the core claim. If **≤2 of 6** pass, the architecture needs redesign or the task needs reformulation.
