# Resonance Transformers: Web-Scale & Structural Task Report

**Date**: 2026-04-30
**Machines**: Nextop (Apple M2 Max, MPS) | Persvati (AMD Ryzen AI 9 HX PRO 370, ROCm 6.2)
**Dataset**: HuggingFaceFW/fineweb-edu (sample-10BT) with GPT-2 BPE tokenization

---

## 1. Executive Summary

We completed four production web-scale language-modeling runs and launched a focused structural-task sweep plus matched 100M-token persvati runs. The headline finding:

> **Standard ALiBi consistently outperforms phase-dynamic on language modeling across both platforms**, but the gap narrows with more training tokens. Whether the phase stream provides a structural inductive bias on controlled tasks is the active question.

**Why no ALiBi + Phase hybrid?** This is the most obvious gap. The current `configure_condition` sets `attention_variant = "alibi"` only for standard configs; resonance configs always use learned sinusoidal or RoPE-style position encodings. An ALiBi+Phase condition would disentangle "does the phase stream hurt because it loses positional bias?" from "is the phase mechanism itself weak?"

---

## 2. Web-Scale Language Modeling Results

### 2.1 Setup

| Hyperparameter | Value |
|---|---|
| Scale | 20M (~62M params with GPT-2 vocab) |
| Architecture | d=512, L=4, H=8, ff=1536–2048, dropout=0.1 |
| Seq len | 512 |
| Batch size | 16 |
| LR | 3e-4 |
| Warmup | 1000 steps |
| Grad clip | 1.0 |
| Optimizer | AdamW (β=(0.9, 0.95), wd=0.1) |
| Schedule | Linear warmup → cosine decay |
| Eval frequency | Every 2000 steps |
| Tokenizer | GPT-2 BPE (vocab=50257, pad=eos) |

### 2.2 Completed Runs

| # | Machine | Backend | Condition | Tokens Trained | Steps | Final Val Loss | Final Val PPL | Avg Tok/s | Checkpoint |
|---|---------|---------|-----------|---------------|-------|---------------|--------------|-----------|------------|
| 1 | Nextop | MPS | `standard_alibi` | ~78.8M | 9,614 | 4.835 | **125.86** | ~10,000 | `std_alibi_20M_s42.pt` (140MB) |
| 2 | Nextop | MPS | `phase_dynamic_qk_film` | ~78.8M | 9,614 | 5.084 | **161.36** | ~4,500* | `phase_dyn_20M_s42.pt` (148MB) |
| 3 | Persvati | ROCm | `standard_alibi` | ~39.8M | 4,862 | 5.257 | **191.83** | ~5,000 | `std_alibi_20M_s42.pt` (141MB) |
| 4 | Persvati | ROCm | `phase_dynamic_qk_film` | ~39.8M | 4,862 | 5.557 | **259.14** | ~4,500 | `phase_dyn_20M_s42.pt` (148MB) |

\* Nextop phase_dynamic used `torch.mps.synchronize()` after every optimizer step to prevent an MPS GPU→CPU sync deadlock. This reduced throughput from ~10K to ~4.5K tok/s but ensured stability.

### 2.3 Loss Curves

**Nextop standard_alibi**
| Step | Val Loss | Val PPL | LR |
|-----:|---------:|--------:|---:|
| 2,000 | 5.785 | 325.3 | 2.94e-4 |
| 4,000 | 5.331 | 206.7 | 2.50e-4 |
| 6,000 | 5.080 | 161.0 | 1.75e-4 |
| 8,000 | 4.905 | 134.6 | 9.28e-5 |
| 9,614 | 4.835 | **125.9** | 3.79e-5 |

**Nextop phase_dynamic_qk_film**
| Step | Val Loss | Val PPL | LR |
|-----:|---------:|--------:|---:|
| 2,000 | 6.144 | 465.9 | 2.94e-4 |
| 4,000 | 5.635 | 279.9 | 2.50e-4 |
| 6,000 | 5.375 | 215.8 | 1.75e-4 |
| 8,000 | 5.173 | 176.5 | 9.28e-5 |
| 9,614 | 5.084 | **161.4** | 3.79e-5 |

**Persvati standard_alibi**
| Step | Val Loss | Val PPL | LR |
|-----:|---------:|--------:|---:|
| 2,000 | 5.773 | 321.5 | 2.72e-4 |
| 4,000 | 5.336 | 207.8 | 1.09e-4 |
| 4,862 | 5.257 | **191.8** | 4.17e-5 |

**Persvati phase_dynamic_qk_film**
| Step | Val Loss | Val PPL | LR |
|-----:|---------:|--------:|---:|
| 2,000 | 6.143 | 465.4 | 2.72e-4 |
| 4,000 | 5.650 | 284.3 | 1.09e-4 |
| 4,862 | 5.557 | **259.1** | 4.17e-5 |

### 2.4 Observations

1. **std_alibi > phase_dynamic on both platforms** by ~28–35% PPL.
2. **Nextop achieves better absolute PPL**, but trained on ~2× more tokens. The Persvati runs were capped at 50M tokens (prefetch buffer limit) vs Nextop's 100M target.
3. **Loss continues to improve** at end of training for all runs; none have saturated.
4. **The gap shrinks with more tokens**: Nextop std_alibi at 40M tokens (step ~4000) had PPL ≈207, while Persvati std_alibi at 40M had PPL ≈208 — nearly identical. This suggests platform differences are small when token counts are matched.

### 2.5 Active / Queued Web-Scale Runs

| Machine | Condition | Target Tokens | Status | Notes |
|---------|-----------|--------------|--------|-------|
| Persvati | `standard_alibi` | 100M | **RUNNING** (PID 1083197) | 78.8M buffer loaded, training started |
| Persvati | `phase_dynamic_qk_film` | 100M | **QUEUED** (auto-launcher PID 1083204) | Will start when std_alibi finishes |

---

## 3. Bugs & Fixes

### Bug 1: MPS Deadlock on `loss.item()`
- **Symptom**: Nextop phase_dynamic hung indefinitely during validation. `sample` showed the main thread stuck in `_local_scalar_dense_mps` → `mps_copy_` → `dispatch_sync_with_rethrow`.
- **Root cause**: MPS command buffer corruption after certain GPU operations. Calling `.item()` on an MPS tensor triggers a GPU→CPU sync that never returns.
- **Fix**: 
  - Added `torch.mps.synchronize()` after every `optimizer.step()` and before/after validation.
  - Rewrote `evaluate()` to accumulate loss as a GPU tensor and call `.item()` only once at the end.
- **File**: `resonance/train_webscale.py`

### Bug 2: `TypeError` in `estimate_params()`
- **Symptom**: `train_webscale.py` crashed on startup with `TypeError: estimate_params() got an unexpected keyword argument 'dropout'`.
- **Fix**: Removed `dropout` from kwargs passed to `estimate_params()`.

### Bug 3: Model Output Dict Handling
- **Symptom**: `AttributeError: 'dict' object has no attribute 'view'` because `ResonanceTransformer` returns `{"logits": ..., "loss": ...}` but training loop expected a raw tensor.
- **Fix**: Added `out["logits"] if isinstance(out, dict) else out` in training and eval loops.

### Bug 4: Prefetch Buffer Loaded 0 Sequences
- **Symptom**: Dataset loaded 0 sequences because `GPT2Tokenizer.encode()` was truncating to `max_length` before the sliding window could chunk long texts.
- **Fix**: Removed `max_length` truncation from `encode()`; let the sliding window handle chunking.

---

## 4. Structural Task Sweep (Active)

### 4.1 Setup
- **Script**: `scripts/run_focused_structural_nextop.sh`
- **Device**: MPS
- **Model**: embed_dim=160, layers=4, heads=4, ff_dim=640
- **Epochs**: 6
- **Seeds**: 1 (701) — initial sweep for speed
- **Tasks**:
  1. cap_matching_depth6 (max_length=448)
  2. algebraic_protocol_depth4 (max_length=640)
  3. unification_depth5 (max_length=256)
  4. cogs_gen (max_length=768, 2500/1000 train/val)
  5. equibench_oj_va (max_length=1536, 300/100)
  6. hans_evaluation (max_length=384, 2500/1000)
- **Conditions** (10 total):
  - Baselines: `standard`, `standard_iso`, `standard_alibi`, `standard_deberta_lite`
  - Phase variants: `phase_stream_only_normalized`, `phase_stream_only_normalized_phase_contrastive`, `resonance_full_normalized`, `phase_dynamic_qk_film`
  - Novel: `complex_directional_normalized`, `relational_stream_lite`

### 4.2 Preliminary Results — cap_matching_depth6

| Condition | Final Val Acc | Final Val Loss | Notes |
|-----------|--------------|---------------|-------|
| `standard` | 0.742 | 0.497 | — |
| `standard_iso` | 0.772 | 0.468 | Best so far |
| `standard_alibi` | 0.728 | 0.494 | Slightly underperforms iso |
| `standard_deberta_lite` | 0.761 | 0.509 | — |

*Sweep is still running (currently on phase conditions for cap_matching). Full results expected in ~2 hours.*

---

## 5. Gaps & Recommendations

### 5.1 Missing Hybrid Conditions
The most important unrun condition is **ALiBi + Phase Dynamic** (or DeBERTa-lite + Phase Dynamic). The current code forces a choice: either you get ALiBi/DeBERTa positional bias *or* you get the phase stream, but not both. This makes it impossible to tell whether phase_dynamic's worse LM performance is:
- (a) the phase mechanism itself being weak, or
- (b) the loss of strong positional bias when switching from ALiBi to learned positions.

**Recommendation**: Add `phase_dynamic_qk_film_alibi` and `phase_dynamic_qk_film_deberta_lite` to `configure_condition`.

### 5.2 Token Count Mismatch
Persvati's first two runs only trained on ~39.8M tokens because the prefetch buffer loaded fewer examples than expected. The active 100M runs will fix this.

### 5.3 No Periodic Checkpointing
`train_webscale.py` only saves at the end of training. If a process crashes or is killed, all progress is lost.

**Recommendation**: Add `--save_every N` to save checkpoints mid-training.

### 5.4 Single-Seed Structural Sweep
The current focused sweep uses 1 seed. If we see promising signals, we should confirm with 3 seeds before drawing architecture conclusions.

### 5.5 MPS Sync Overhead
The `torch.mps.synchronize()` fix costs ~50% throughput. A less aggressive sync strategy (e.g., every 10 steps instead of every step) might recover speed while still preventing hangs.

---

## 6. Checkpoints Available

| Checkpoint | Location | Size | Description |
|-----------|----------|------|-------------|
| `std_alibi_20M_s42.pt` | Nextop `outputs/webscale/` | 140MB | 78.8M tokens, ppl=125.9 |
| `phase_dyn_20M_s42.pt` | Nextop `outputs/webscale/` | 148MB | 78.8M tokens, ppl=161.4 |
| `std_alibi_20M_s42.pt` | Persvati `outputs/webscale/` | 141MB | 39.8M tokens, ppl=191.8 |
| `phase_dyn_20M_s42.pt` | Persvati `outputs/webscale/` | 148MB | 39.8M tokens, ppl=259.1 |

All checkpoints contain `model_state_dict`, `config`, `history`, `tokens_seen`, and `step`.

---

## 7. Active Processes Summary

| Machine | PID | Type | Status | CPU | Elapsed |
|---------|-----|------|--------|-----|---------|
| Nextop | 83491 | Structural sweep (cap_matching) | RUNNING | ~33% | ~7m |
| Persvati | 1083197 | Web-scale std_alibi 100M | RUNNING | ~96% | ~1m |
| Persvati | 1083204 | Auto-launcher (phase_dynamic 100M) | SLEEPING | — | — |

---

*Report generated from live logs and checkpoint metadata. Structural results are preliminary.*
