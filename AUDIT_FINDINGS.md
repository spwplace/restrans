# Experiment audit — correctness & conceptual review

**Date:** 2026-05-31. **Auditors:** 5 parallel review agents (core kernels, cap/algebraic
tasks, automata/lambda/vm tasks, eval harness, trainers+writeups), each grounded in
`research/methodology.md` (iso-param faithful baselines, no train/val leakage, claims must
not outrun the code). All findings below were verified by **running the code / generating
data / re-deriving math**, not by reading alone.

## TL;DR

Good news: **no fabricated numbers were found** — the writeups (esp. `WALKFORMER_FINDINGS.md`)
are unusually candid and often self-caveat. The problems are upstream of the prose: two
**comparison-invalidating** harness defaults (non-iso baseline + tokenizer fit on val), three
**leakage/trivial-solvability** bugs in task generators that let a one-line heuristic "solve"
the task, and a couple of **conceptual over-claims** in the kernel (the "chiral" atom isn't
actually directional). Fix these and the harness produces trustworthy comparisons; the
mechanisms themselves (Bessel CTQW math, causal masking, ALiBi slopes, RLVR↔plan
correspondence) are sound.

Severity key: **CRITICAL** = invalidates a result/comparison · **BUG** = wrong but recoverable
· **SMELL** = quality/clarity.

---

## CRITICAL — fix before citing any comparison

### C1. Baselines are not iso-parameter in almost every run script
`structural_task_probe.py` default conditions + nearly all `scripts/run_*.sh` compare
resonance/walkformer variants against bare `standard` / `standard_alibi`. Measured:
`standard` = **1,348,352** params; every resonance variant = **1,483,664** (**+10%**, +135k
active phase params). The iso machinery exists and is correct (`standard_iso` = 1,484,848 ≈
walkformer 1,483,718, within 0.08%) — the scripts just don't use it.
**Fix:** make `standard_iso` / `standard_iso_alibi` the baseline everywhere; error if a bare
`standard*` baseline is mixed with resonance conditions.

### C2. Tokenizer vocabulary is fit on validation text (val leakage)
All four harnesses call `tokenizer.train(train_ds.all_texts() + val_ds.all_texts() + …)`
(`structural_task_probe.py:330`, `regime_probe.py:533`, `story_query_eval.py:532`,
`story_topology_eval.py:923`). With small `--vocab_size` (512/768) this changes which val
tokens are `<UNK>`. **Fix:** train tokenizer on train text only (+ the fixed `yes`/`no`
answer tokens).

### C3. Webscale train/val sets overlap
`train_webscale.py` builds both `FinewebStreamingDataset` from `split="train"` from doc 0 with
no skip/shuffle — val is a strict subset of train. Reported val PPLs (125.9 / 161.4 / 191.8 /
259.1) are partly measured on training docs. **Fix:** give val a disjoint slice
(`ds.skip(train_prefetch)` or a held-out shard). Also: the two arms compare mismatched token
budgets (Nextop 78.8M vs Persvati 39.8M), so the "gap shrinks with tokens" sub-claim isn't
supported.

### C4. `algebraic_protocol` negatives leak the answer via out-of-knowledge atoms
`algebraic_protocol.py:101-108`: positives are drawn from `closure(initial)` (built from the
3–5 sampled atoms); negatives draw from the **full 8-atom** vocabulary. The rule "answer NO if
the query contains any atom absent from the known-facts text" scores **96.8%** (majority 50%).
No Dolev-Yao reasoning required. Closure semantics are otherwise correct. **Fix:** restrict
negative atom pool to the atoms used in `initial`; prefer hard (non-derivable-over-known)
negatives.

### C5. `vm_tasks` equivalence task is 100% prefix-solvable
`vm_tasks.py:308-327`: positives are `right = list(left) + [DUP, POP]` (or `[PUSH 0, POP]`) —
B is literally A's text plus two identity ops. Predictor "B starts with A" = **100%**;
"#ops(B)==#ops(A)+2" = **100%**. The VM is never executed. (VM *step* and *trace* tasks are
clean — 0/600 mislabels.) **Fix:** build positives as genuinely different programs verified to
reach the same final stack.

### C6. Cascade metric leaks the gold terminal flag
`train_structural_posttrain.py:271-276`: `cascade_acc`'s `terminal_ok` is read straight from
the verifier metadata (`terminal_normal` / `terminal_matches`), i.e. the second conjunct of the
ground-truth label — not predicted by the model. For `wrong_terminal` negatives the model is
never consulted. Absolute cascade numbers (lambda 0.633 vs 0.553; VM 0.731 vs 0.572 in
`cascade_posttraining_report.md`) are inflated; between-condition deltas may survive but rest on
a contaminated metric. **Fix:** predict the terminal with the model, or report local-step
composition only.

---

## BUG — wrong, recoverable, confounds comparisons

### B1. lambda equivalence: severe length confound
`lambda_tasks.py:159-163`: `_mutated_equivalent` positives are bloated (len(right) ≈ **80.7**
vs negatives **23.9**). Threshold `len(right)>39` predicts label at **62.5%**; "contains
'lambda'" 73% pos vs 51% neg. **Fix:** cap/size-match mutated positives to the negative length
distribution.

### B2. lambda beta-step / trace: fallback negatives can equal the correct answer
`lambda_tasks.py:268-271` and `:360-362`: when a "negative" coincidentally equals the true
next step / normal form, the code substitutes a random term and force-sets `valid=False`
**without rechecking** — so the replacement can itself be correct (≈0.3% of negatives; concrete
seed-97 example: `(λx. x) false` → `false` labeled "no"). **Fix:** recompute `valid` after the
fallback substitution and resample until it differs.

### B3. lambda equivalence: ~5.5% of negatives silently flip to positive + degenerate NF space
`lambda_tasks.py:195-197`: the depth-1 fallback often produces `()`/a constant whose NF equals
`left_nf`, making `equivalent=True`. Labels stay honest but balance drifts (419 yes / 381 no
over 800). Compounded: **53%** of all terms reduce to a trivial constant. **Fix:** resample
fallback until its NF differs; bias generator away from constant-typed terms.

### B4. automata: iso-vs-perturbation design is count-separable
`automata_tasks.py`: positives = state-renaming isomorphisms (preserve accept-count &
self-loop-count exactly); negatives = local perturbations (usually change them). Predictor
"(accept-count, self-loop-count) match" = **81.3%**. Labels themselves are exact (0/400
mismatch). **Fix:** generate negatives that preserve surface-invariant statistics; add
non-isomorphic-but-equivalent positives.

### B5. cap_matching: positive generator mislabels ~7% of intended positives
`cap_matching.py:246-257`: `mask_target` can place the same variable name over two *different*
subterms; `match_cap` correctly rejects these, so ~7% of intended positives become negatives
(pos_frac 0.463 vs 0.50). **Fix:** memoize var→subterm so a reused variable binds one subterm,
or use fresh names per masked position.

### B6. cap_matching: fuel exhaustion fabricates a label
`cap_matching.py:121-122` `if fuel <= 0: return True` (and `match_cap` returns `None`) — claims
intersection on budget exhaustion. Latent at default depth (guard never fires) but will flip
labels in the advertised depth-extrapolation configs (`depth5_to7`, `depth7`). **Fix:** drop
the example on exhaustion instead of fabricating `True`/`None`; or scale fuel to term size.

### B7. "chiral" atom is symmetric — not direction-aware
`kernels.py:507-523`: the headline chiral atom computes `cos(φ·d)·decay(|d|)`, both factors
even in `d` ⇒ `R[i,j]==R[j,i]` (verified `allclose(R,R.T)=True`). The docstring's "direction-
aware band that real SSMs/Hyena/FNet can't express" is false — `Re(e^{iφd})=cos(φd)` discards
exactly the direction. **Fix (this is an *upgrade*, makes the idea real):** add the
antisymmetric `sin(φ·d)` component or a learnable sign-of-`d` gain (as the `coined` atom
already does). Note: this changes the walkformer numbers and should be re-run.

### B8. Toeplitz / length-extrapolation claim broken under default `walk_use_phase_drive=True`
`kernels.py:507-517`: with phase-drive on (the default, `config.py:71`), atom-4's angle depends
on per-token mean phase ⇒ not a function of `i-j` only (two d=1 entries: 0.589 vs 0.0085). The
"every atom Toeplitz / length-extrapolating" claim holds only with phase-drive off. Causally
safe. **Fix:** gate phase-drive off for extrapolation claims, or scope the docstring.

---

## SMELL — quality / clarity (non-blocking)

- **ALiBi+resonance isn't iso-information vs plain ALiBi** (`models.py:687-721`): the resonance
  condition gets ALiBi *and* the learned bias; compare only against ALiBi+zeroed-bias control.
- **`eval_blimp.py`** (`:58-84`): no UNK guard (pairs decided by noise still counted), no
  length-normalization option (biases length-differing paradigms), `len<2 → 0.0` compares
  real loglik to 0, strict `>` scores ties as wrong.
- **No held-out test split** anywhere — all headline numbers are final-epoch *val* (no
  select-best-on-val bug, but val is doing double duty). Add a third split.
- **`learnedH` atom** uses circular mod-S offset on a graph documented as a path
  (`kernels.py:638`) — length-dependent, undercuts extrapolation.
- **`acc_gain = after − before`** in posttrain (`:947`) is measured from random init, so it's
  "did it learn the task at all," not RLVR lift.
- **`grpo_enum`** over a binary enumerated group with std-normalized advantage is degenerate
  GRPO — fine as labeled, but it's not group-sampled RL.
- **`ignore_index=50256`** in webscale (`:303,389`) silently drops real EOS labels (pad==eos);
  no padding actually exists in the dense chunks.
- **`process_score` / `valid_local_steps` metadata is a perfect label oracle** for trace tasks
  — harmless unless it ever enters model input; worth a guard comment.

---

## What's actually SOUND (verified — don't re-litigate)

- Bessel CTQW atom: `J_d(2τ)` via Jacobi-Anger/FFT matches the series exactly incl. parity
  `J_{-d}=(-1)^d J_d`. Finite fwd+grad across all atom sets.
- Causal masking correct end-to-end including the structural-head overwrite path.
- ALiBi slopes are a faithful Press et al. construction (n=8 → 0.5,0.25,…).
- `WALKFORMER_FINDINGS.md` numbers match `outputs/walkformer_gauntlet/.../results.json` exactly,
  iso-param, honestly caveated as within-seed-noise.
- RLVR plan ↔ `train_structural_posttrain.py` correspond (sft/dpo/exact_rl/grpo, frozen ref,
  curriculum, verifier rewards, ablation diagnostics).
- VM step & trace tasks, automata labels, cap_matching surface-baseline gap (~0.64 probe vs
  ~0.73–0.77 model) are sound. No exact train/val *text* overlap in synthetic tasks
  (seed offset +100k).

---

## Open design decisions (need Ember's call — change what experiments measure)

1. **Hard-negative redesigns** (C5 vm-equivalence, B4 automata, B1/B3 lambda-equivalence):
   these change the data distribution and invalidate any in-flight runs on those tasks. Redesign
   now, or freeze those tasks and lean on the sound ones (vm step/trace, cap_matching,
   unification)?
2. **Chiral atom fix (B7):** making it directional is the *right* fix but changes walkformer
   results. Re-run the gauntlet after?
3. **Iso-baseline rollout (C1):** flip every script to `standard_iso*`, or also keep bare
   `standard` runs for backward-comparability with logged results?
