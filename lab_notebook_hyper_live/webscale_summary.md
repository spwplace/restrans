# Web-Scale LM Training Results

Date: 2026-04-30

## Setup
- Dataset: HuggingFaceFW/fineweb-edu (sample-10BT)
- Tokenizer: GPT-2 BPE (vocab=50257)
- Model scale: 20M (~62M params with GPT-2 vocab)
- Config: d=512, L=4, H=8, ff=1536 (or 2048), dropout=0.1
- Seq len: 512, Batch: 16
- LR: 3e-4, warmup: 1000 steps, grad clip: 1.0

## Results

| Machine | Condition | Tokens | Final Val Loss | Final Val PPL | Steps | Tok/s |
|---|---|---:|---:|---:|---:|---:|
| Nextop (MPS) | standard_alibi | ~78.8M | — | 125.86 | ~6103 | ~10K |
| Nextop (MPS) | phase_dynamic_qk_film | ~78.8M | 5.0836 | 161.36 | 9614 | ~4.5K* |
| Persvati (ROCm) | standard_alibi | ~39.8M | 5.2566 | 191.83 | 4862 | ~5K |
| Persvati (ROCm) | phase_dynamic_qk_film | ~39.8M | 5.5574 | 259.14 | 4862 | ~4.5K |

\* Nextop phase_dynamic used `torch.mps.synchronize()` after each step for stability, 
which reduced throughput from ~10K to ~4.5K tok/s.

## Observations
1. **std_alibi consistently outperforms phase_dynamic** on both platforms:
   - Nextop: 125.86 vs 161.36 PPL (~28% worse)
   - Persvati: 191.83 vs 259.14 PPL (~35% worse)
2. **Nextop achieves better absolute PPL** than Persvati, but trained on ~2x more tokens.
   Persvati runs were capped at 50M tokens vs Nextop's 100M.
3. **Training was stable** after MPS sync fixes (added `torch.mps.synchronize()` 
   after optimizer steps and around validation).
4. **Loss curves** (Nextop phase_dynamic): step 2000→6.14, 4000→5.64, 6000→5.37, 8000→5.17, final→5.08
5. **Loss curves** (Persvati std_alibi): step 2000→5.77, 4000→5.34, final→5.26
6. **Loss curves** (Persvati phase_dynamic): step 2000→6.14, 4000→5.65, final→5.56

## Next Steps
- Re-run Persvati with matched 100M token budget for fair comparison
- Test additional conditions (resonance_full, phase_stream_only, relational_stream_lite)
- Scale up to 50M parameter models
- Analyze whether phase_dynamic's worse LM performance correlates with structural-task gains
