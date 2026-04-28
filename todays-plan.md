# Today's Plan: Structural Task Pivot

Date: 2026-04-27

## Goal

Move from microvalidation as the main activity to a literature-task-centered
experiment programme.  Microvalidations remain as indicators and debugging
checks, not as results we optimize for.

## Execution Order

1. **Expand the dataset lake**
   - Keep the already-downloaded core set:
     - WikiText-2 / WikiText-103,
     - TinyStories subset by default, full TinyStories only with
       `--tinystories_full`,
     - BLiMP,
     - COGS,
     - public AMR derivative,
     - generated Linzen-style agreement.
   - Add the structural/literature sources now available:
     - BabyLM strict-small for fast iteration,
     - full BabyLM snapshot when explicitly requested with `--babylm_full`,
     - Linzen `rnn_agreement`,
     - SLOG,
     - HANS,
     - MSGS,
     - SCAN,
     - CFQ MCD splits,
     - CodeXGLUE POJ-104.
   - Add larger code datasets only when disk/time is acceptable:
     - CodeSearchNet,
     - BigCloneBench.
   - Treat `--optional_export_limit` as a convenience-export cap only.  It does
     not truncate the saved Hugging Face dataset; it keeps giant JSONL mirrors
     from becoming the bottleneck.

2. **Lock the shared task harness**
   - Use `resonance/structural_task_probe.py` for all yes/no structural probes.
   - Keep `resonance/regime_probe.py` as the common training and geometry gate.
   - Smoke every task at tiny scale after edits.

3. **Run the primary 10M-ish syntax experiment**
   - Start with `agreement`, then replace with local Linzen/BLiMP data when
     downloaded.
   - Conditions:
     - `standard`
     - `resonance_full_normalized`
     - `bias_only_normalized`
   - Suggested first real run:

```bash
.venv/bin/python resonance/structural_task_probe.py \
  --task agreement \
  --output_dir resonance/outputs/agreement_10m_probe \
  --epochs 10 \
  --train_examples 20000 \
  --val_examples 4000 \
  --max_attractors 5 \
  --embed_dim 352 \
  --layers 6 \
  --heads 8 \
  --ff_dim 1408 \
  --batch_size 48 \
  --conditions standard,resonance_full_normalized,bias_only_normalized \
  --seeds 11 23 37 \
  --eval_each_epoch
```

4. **Add established local data**
   - BLiMP, once present locally:

```bash
.venv/bin/python resonance/structural_task_probe.py \
  --task blimp \
  --data_path /path/to/blimp/paradigm.jsonl \
  --output_dir resonance/outputs/blimp_probe \
  --epochs 5 \
  --train_examples 5000 \
  --val_examples 1000 \
  --conditions standard,resonance_full_normalized,bias_only_normalized
```

   - Linzen-style agreement file, once present locally:

```bash
.venv/bin/python resonance/structural_task_probe.py \
  --task agreement_file \
  --data_path /path/to/agreement.jsonl \
  --output_dir resonance/outputs/linzen_agreement_probe \
  --epochs 5 \
  --train_examples 20000 \
  --val_examples 4000 \
  --conditions standard,resonance_full_normalized,bias_only_normalized
```

5. **Run synthetic indicators, not claims**
   - Use these to see whether a model trained for syntax has structural
     side-effects, or to diagnose task failures:
     - `template_equivalence`
     - `dyck`
     - `unification`
     - `graph_alias`
     - `causal_intervention`
     - `structural_paraphrase`

6. **Rank tasks by ablation-readiness**
   - Candidate only if it satisfies all three:
     - accuracy beats majority by at least 5 points;
     - validation loss improves;
     - label geometry separates (`label_gap > 0`).
   - Saturated tasks stay as smoke tests.
   - Chance tasks become generator/debugging tasks.

7. **Only then run iso-parameter ablations**
   - For the first task that passes the gate, run:
     - standard iso-param;
     - resonance full;
     - phase stream only;
     - bias only;
     - inert resonance;
     - explicit relation-aware baseline if applicable.

## Today's Concrete Success Criteria

- Dataset manifest includes the new sources that are locally available.
- All new task modules compile.
- The generic probe can run one tiny smoke pass for each synthetic task.
- We have one primary command for the 10M agreement direction.
- We have local-file adapters ready for BLiMP and Linzen-style agreement data.

## Interpretability Integration

- Treat built-in sparsity as an experimental condition, not a post-hoc cleanup:
  add sparse structural bottlenecks and compare against matched standard
  sparse-bottleneck controls.
- Treat geometry as a first-class output:
  SAEs are useful, but also fit low-rank, convex-archetype, and spectral probes
  to phase/resonance streams.
- Report interpretability only when it is causal:
  the feature/probe must survive patching, ablation, or steering tests.

## Interpretation Discipline

Do not report microtask wins as evidence for the architecture.  A microtask can
only say one of:

- this regime is saturated;
- this regime is too hard or misgenerated;
- this regime is a candidate for larger ablation;
- this trained model exposes useful structural diagnostics.
