# Cap Matching Integration

The cap-matching line is now integrated as a reproducible synthetic formal
workstream.

## Source And Reproducibility

Reference implementation:

- `https://github.com/emberian/reu_unif.git`
- pinned revision: `c0da84c2d8d392031ab252e3617d43a9c9de96d0`
- setup command: `bash scripts/setup_external_deps.sh`

The Rust reference currently builds its library on a modern toolchain but the
binary uses obsolete nightly-era APIs. For training we therefore use
self-contained Python fragments in this repo, with the Rust source treated as a
reference artifact rather than a runtime dependency.

## Implemented Training Tasks

### `cap_matching`

File: `resonance/synthetic/cap_matching.py`

Decision problem:

```text
Given pattern term s and Cap object T, does s match at least one ground term
represented by T?
```

This is a deliberately smaller label than the full paper result, which returns a
complete finite representation of all Cap-substitutions. The probe still tests:

- variables and repeated-variable consistency,
- function-symbol decomposition,
- union,
- unbounded constructor closure through `Cap{...}(...)`,
- nested Cap objects under ordinary constructors.

The solver avoids enumerating the Cap closure. It recursively decides whether a
finite pattern can be generated from the Cap object.

### `algebraic_protocol`

File: `resonance/synthetic/algebraic_protocol.py`

Decision problem:

```text
Given initial intruder knowledge and algebraic abilities, can the intruder
derive a query message?
```

Abilities:

- construct pair `p(x,y)`,
- construct encryption `e(m,k)`,
- split pair `p(x,y) -> x,y`,
- decrypt `e(m,k)` when `k` is known.

This is a bounded Dolev-Yao-style reachability task. It is intended to test the
same algebraic structure from another angle: not just matching a pattern
against a Cap object, but computing closure under constructor/destructor rules.

## Example Commands

```bash
export PYTHONPATH=resonance

uv run python resonance/structural_task_probe.py \
  --task cap_matching \
  --output_dir resonance/outputs/cap_matching_probe \
  --conditions standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized \
  --seeds 1 2 3 \
  --epochs 4 \
  --train_examples 1500 \
  --val_examples 500 \
  --depth 5 \
  --device mps

uv run python resonance/structural_task_probe.py \
  --task algebraic_protocol \
  --output_dir resonance/outputs/algebraic_protocol_probe \
  --conditions standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized \
  --seeds 1 2 3 \
  --epochs 4 \
  --train_examples 1500 \
  --val_examples 500 \
  --depth 3 \
  --device mps
```

## Current Hypothesis

These tasks should be stronger tests of the phase/structural channel than the
earlier random graph and toy beta-reduction tasks because the labels are tied to
explicit algebraic structure. A useful architecture signal would be:

1. phase/full variants beat `standard`,
2. the win survives `standard_iso`,
3. the win moves validation loss and label geometry, not just accuracy,
4. full resonance separates from phase-only on at least one algebraic task.

If `standard_iso` wins, the result is still useful: it means the task is
capturing real structure, but the current phase mechanism is not yet the right
inductive bias.
