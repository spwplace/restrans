# Resonance Transformer — Training Guide

## Quick Start

All training goes through the unified harness:

```bash
uv run python resonance/train.py --mode <MODE> --scale <SCALE> [options]
```

### Modes
- `standard`   — Vanilla transformer on structured synthetic data
- `resonance`  — Resonance transformer (dual embedding + phase bias)
- `synthetic`  — Contrastive proof-walk training on lambda calculus

### Scales
| Scale  | Epochs | Samples/Groups | Model dim | Layers | Batch | Notes |
|--------|--------|----------------|-----------|--------|-------|-------|
| tiny   | 5      | 1K / 200       | 64–128    | 2–4    | 8–16  | Smoke test |
| small  | 20     | 10K / 2K       | 128–256   | 4–6    | 16–32 | Dev loop |
| medium | 30     | 50K / 10K      | 256–512   | 6–8    | 32    | Meaningful |
| large  | 50     | 200K / 50K     | 512–768   | 8–12   | 16–32 | Serious |

## Apple Silicon (MPS)

MPS is auto-detected. No flags needed:

```bash
uv run python resonance/train.py --mode resonance --scale medium
```

To force a device:

```bash
uv run python resonance/train.py --mode standard --scale small --device mps
```

## Synthetic Proof-Walk Training

The generator now supports three crawl modes:

- **structured** — Enumerates statements by type complexity and context size
- **random** — Pure random sampling with diverse term shapes
- **hybrid** *(default)* — Interleaves structured and random for coverage + diversity

```bash
uv run python resonance/train.py --mode synthetic --scale medium --crawl_mode hybrid
```

With scheduled bisimilar mutations:

```bash
uv run python resonance/train.py --mode synthetic --scale medium --mutate
```

## Reproducibility

Every run saves:
- `config.json` — full hyperparameters and scale preset
- `reproducibility.json` — seed, PyTorch version, device
- `dataset_state.json` — generator state (for exact regeneration)
- Checkpoints every N epochs + final model

To reproduce exactly:

```bash
uv run python resonance/train.py --mode resonance --scale small --seed 42
```

## Outputs

All results land in `resonance/outputs/<mode>_<scale>/`:

```
outputs/resonance_small/
  config.json
  reproducibility.json
  history.json
  checkpoints/
    final.pt
    epoch_5.pt
    ...
```

## Legacy Scripts

The older scripts in `scripts/` still work but do not have the new scale presets or unified MPS handling:

```bash
uv run python resonance/scripts/train_single.py --model resonance --name my_run --device mps
uv run python resonance/scripts/train_synthetic.py --epochs 20 --groups 1000 --device mps
```
