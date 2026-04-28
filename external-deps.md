# External Dependencies

This repository should be reproducible from a fresh clone. Large datasets and
research-code dependencies are fetched into gitignored local directories rather
than assumed to exist on one machine.

## Cap Matching / Knowledge-Based Unification

The cap-matching source used for reference lives at:

- Repository: `https://github.com/emberian/reu_unif.git`
- Reference revision observed locally: `c0da84c2d8d392031ab252e3617d43a9c9de96d0`
- Default checkout path: `external/reu_unif`

Fetch it with:

```bash
bash scripts/setup_external_deps.sh
```

The original Rust implementation is treated as a reference artifact. The
experiment harness should not depend on a local `~/dev/reu_unif` path. Any
dataset generators used in this repo should either:

1. call `external/reu_unif` through a documented wrapper, or
2. include a self-contained Python implementation of the exact fragment used by
   the experiment.

For now we use option 2 for training probes, while preserving option 1 for
trace/debug comparison against the original implementation.

## Data Lake

Datasets are staged under `data/`, which is intentionally gitignored. See
`dataset-lake.md` for the current download commands and license notes.
