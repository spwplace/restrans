# Resonance Transformer: Methodology for Rigorous Evaluation

> *"What do we even have, and how does it compare?"*

## 1. What We Have (Architecture Audit)

### 1.1 Phase Embedding: How It Learns

The `phase` embedding is **not hand-engineered**. It is a standard `nn.Embedding(vocab_size, n_frequencies)` table that learns from gradient descent just like any other parameter.

**Forward-path gradients to `phase.weight`:**

```
Path A (representation):  phase → phase_proj → blend → blocks → loss
Path B (attention bias):  phase → get_resonance_matrix() → attn bias → blocks → loss
```

- **Path A** treats phase as an auxiliary representation stream. After projection to `embed_dim`, it is blended with the semantic stream via a learnable per-dimension sigmoid gate (`blend`).
- **Path B** is the novel one. Pairwise phase differences are converted to cosine similarities, producing a resonance matrix `R[i,j] = mean_f cos(φ_i^f − φ_j^f)`. This matrix is added as a bias to attention logits before softmax: `attn = QK^T/√d + R · w_r^(h)`.

Because Path B creates a **differentiable pairwise relationship** between every token pair in the sequence, the phase embedding learns to arrange tokens in a continuous space such that structurally or semantically related tokens receive favorable attention biases *before the Q/K projections even matter*. At initialization this is random noise. After training, tokens that should attend to each other move closer in phase space.

**Initialization difference:**
| Parameter | Init std | Rationale |
|-----------|----------|-----------|
| `semantic` | 0.02 | Standard GPT-style |
| `phase` | 0.30 | 15× larger; phases need to span the periodic cosine manifold quickly |
| `blend` | sigmoid(0.3) ≈ 0.57 | Slight preference for phase stream at start |
| `resonance_weight` | 0.1 | Small initial attention bias |

### 1.2 What Tokenizer We Are Using

**Current (`experiment_text.py`):** A custom **word-level tokenizer** (`WordTokenizer`):
- Lowercases text
- Splits on whitespace
- Keeps top 5,000 words by frequency
- Vocab: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>` + 4,996 words
- Sequence length: 128 word tokens

**This is NOT comparable to published TinyStories baselines.**
- **einygpt**: GPT-2 BPE tokenizer (vocab ~50K, subword)
- **TinyStories paper**: GPT-Neo tokenizer or GPT-2 tokenizer
- **TernaryLM**: Custom BPE (vocab 2,048–10,000)

**Implication:** Perplexity is a function of tokenization granularity. Word-level PPL tends to be *higher* than subword PPL because predicting whole words is harder than predicting character chunks. Our 56 PPL on word-level is likely **better than it looks** when compared to BPE baselines, but we cannot make numerical comparisons without conversion.

---

## 2. Internal Validation: Baseline vs Resonance

These experiments answer: *"Does the resonance mechanism itself help, or is it just extra parameters?"*

### 2.1 Current Experiment Design (Running Now)

| Condition | Model | Params | Dataset | Tokenizer |
|-----------|-------|--------|---------|-----------|
| `baseline` | `StandardTransformer` | 5.53M | TinyStories 100K | Word-level 5K |
| `resonance` | `ResonanceTransformer` | 5.70M | TinyStories 100K | Word-level 5K |
| `proof_prior` | `StandardTransformer` | 5.53M | Proof-walks → TinyStories | Word-level 5K |

**Control quality:** ✅ Same tokenizer, same data order (different seeds), same hyperparameters, ~3% param difference.

### 2.2 Required Ablations (Phase 0b)

To isolate the mechanism, run these **with identical seeds and data order**:

1. **`resonance_weight = 0`** — Train `ResonanceTransformer` but zero out the attention bias. If performance drops to baseline, the benefit comes from Path B (attention biasing), not just the extra embedding stream.
2. **`blend = 1.0` (frozen)** — Force pure semantic stream, disable phase projection blending. If performance drops, Path A (dual representation) matters.
3. **`phase_rand_init` vs `phase_phonetic_init`** — Test whether phonetic rhyme-based initialization (already implemented but disabled) accelerates early training.
4. **Same-seed replication** — Run baseline and resonance with `--seed 42` for both to eliminate seed variance.

### 2.3 Statistical Rigor

- **Minimum 3 seeds** per condition for significance testing.
- Report mean ± std for: epoch-1 PPL, epoch-5 PPL, final PPL, wall-clock time.
- Effect size: compute `ΔPPL / σ_baseline` (Cohen's d equivalent).

---

## 3. External Comparability: Matching the Literature

To answer *"How does this compare to einygpt / TinyStories paper / Mamba / etc.?"* we must switch to a **standard tokenizer and standard dataset split**.

### 3.1 Recommended Standardization Run

Create `experiment_text_standardized.py` or add `--tokenizer gpt2` flag:

| Setting | Current | Standardized |
|---------|---------|--------------|
| Tokenizer | Custom word-level 5K | GPT-2 BPE (50,257 vocab) |
| Seq length | 128 words | 256 tokens (BPE) |
| Dataset | 100K stories sample | Full TinyStories train (~2.6M) |
| Val set | 10K stories sample | Official TinyStories validation |
| Model dims | 256d, 4L, 8H | Match einygpt: 384d, 6L, 6H or comparable |

**Why GPT-2 BPE?**
- Most published TinyStories results use it
- Enables direct PPL comparison
- HuggingFace `transformers` provides it trivially

**Parameter-matching rule:**
When comparing to einygpt (4.3M or 6.9M params), adjust `embed_dim` and `n_layers` so that `StandardTransformer` and `ResonanceTransformer` are **iso-parameter** with the external baseline. The resonance version should match the baseline's parameter count by slightly reducing `embed_dim` (e.g., 248d vs 256d) so total params are equal.

### 3.2 Benchmark Targets

| Source | Params | Tokenizer | Reported Val PPL | Our Target |
|--------|--------|-----------|------------------|------------|
| einygpt | 4.3M | Custom BPE | ~1.0* | Train same tokenizer |
| einygpt | 6.9M | GPT-2 BPE | ~1.0* | Train GPT-2 BPE |
| TinyStories-33M | 33M | GPT-2 | ~62 (TernaryLM cites) | Far future |
| TernaryLM FP32 | 132M | Custom | 28.25 | Not our scale |

*einygpt's "1.0" is suspiciously low and may be train PPL or a different metric. Verify by running their code.

### 3.3 When to Standardize

- **Do NOT stop current runs.** They are valid for the internal baseline-vs-resonance comparison.
- **Schedule standardized runs** after Phase 0 completes (i.e., after we confirm the internal effect is real and reproducible).
- **First standardized run:** Match einygpt's 4.3M–6.9M scale with GPT-2 BPE. This is a weekend run on persvati GPU.

---

## 4. Mechanistic Understanding: What Is the Model Actually Doing?

These experiments answer: *"Why does it work?"*

### 4.1 Representation Geometry

| Experiment | What to Measure | Tool |
|-----------|-----------------|------|
| PCA on `semantic` vs `phase` embeddings | Effective rank, variance distribution | `torch.pca_lowrank` |
| t-SNE / UMAP of phase space | Clustering by POS, phonetics, word length | `sklearn.manifold` |
| Cross-predictability | Can a linear probe predict semantic from phase? | Ridge regression |
| Resonance matrix analysis | Sparsity, diagonal pattern, head specialization | Custom analysis |

### 4.2 Intervention Experiments

| Intervention | Expected Result if Phase Matters |
|--------------|----------------------------------|
| Zero out `phase` at inference | PPL should degrade significantly |
| Zero out `semantic` at inference | PPL should degrade catastrophically |
| Freeze `phase` after epoch 1, continue training | If phase is a "structural prior," frozen should still help |
| Swap phase embeddings between two trained models | PPL should degrade (phase is model-specific) |

### 4.3 Attention Visualization

For a given sequence, plot:
- Standard attention weights `softmax(QK^T)`
- Resonance bias matrix `R`
- Combined attention `softmax(QK^T + R)`

Look for cases where resonance "fills in" attention that QK^T misses (e.g., long-distance anaphora, rhyme pairs, repeated phrases).

---

## 5. Scaling & Transfer Roadmap

### 5.1 Scaling Laws (Phase 1)

Train at multiple scales with **iso-parameter matching** between Standard and Resonance:

| Scale | Params | Embed | Layers | Heads | Dataset | Epochs |
|-------|--------|-------|--------|-------|---------|--------|
| Tiny | 1M | 128 | 2 | 4 | TinyStories 100K | 20 |
| Small | 5M | 256 | 4 | 8 | TinyStories 100K | 20 |
| Medium | 20M | 512 | 6 | 8 | TinyStories 1M | 10 |
| Large | 50M | 768 | 8 | 12 | TinyStories full | 5 |

**Key question:** Does the `Resonance / Standard` PPL ratio improve, stay constant, or degrade as scale increases? Many architectural tricks help at small scale but vanish at large scale.

### 5.2 Curriculum Transfer (Phase 1b)

The `proof_prior` condition tests a specific hypothesis: *synthetic structured reasoning → natural language.*

To make this rigorous:
1. **Control condition:** Pre-train baseline on *unstructured* synthetic data (random token sequences of same length/distribution as proof-walks). If proof-walks help more than random sequences, structure matters.
2. **Varying pre-train length:** 1, 5, 10, 20 epochs of proof-walks → measure transfer curve.
3. **Layer-wise transfer:** Freeze lower layers after pre-training, fine-tune only upper layers. If lower layers contain transferable structure, this should work.

### 5.3 Downstream Tasks (Phase 2)

PPL is a proxy. Real validation requires:
- **Story completion:** Given a prompt, generate 100 tokens; score with GPT-4 for grammar, coherence, creativity (TinyStories paper protocol).
- **BLiMP:** Grammaticality judgments (linguistic generalization).
- **HellaSwag / PIQA:** Commonsense reasoning (if scaled up).

---

## 6. Immediate Action Items

### This Week (While Current Runs Finish)
1. ✅ Let baseline + resonance + orchestrator run to completion.
2. ⬜ Add `--tokenizer gpt2` option to `experiment_text.py` for standardized runs.
3. ⬜ Implement `ResonanceTransformer(use_resonance=False)` ablation flag.
4. ⬜ Write analysis script to load checkpoints and compute: PCA rank, phase clustering, resonance matrix stats.

### Next Week (Validation)
5. ⬜ Run same-seed ablation grid: baseline, resonance, resonance_weight=0, blend=1.0 frozen.
6. ⬜ Run 3 seeds for baseline vs resonance; compute mean ± std.
7. ⬜ Launch first standardized run (GPT-2 BPE, einygpt-scale).

### Next Month (Scaling)
8. ⬜ Scaling law sweep (1M → 50M).
9. ⬜ Full TinyStories training with GPT-2 BPE.
10. ⬜ GPT-4 story-completion evaluation.

---

## 7. Summary: Claims We Can Make Now vs Later

| Claim | Evidence Required | Status |
|-------|-------------------|--------|
| "Resonance converges faster on word-level TinyStories" | 1 run, different seeds | ⚠️ Promising, N=1 |
| "Resonance has 3% param overhead" | `count_params()` | ✅ Confirmed |
| "Phase embeddings learn a separate, compressible representation" | PCA + quantization exps | ✅ From earlier experiments |
| "Resonance beats standard transformers on standard benchmarks" | GPT-2 BPE, multiple seeds, iso-param | ❌ Not yet tested |
| "Proof-walk pretraining accelerates language learning" | `proof_prior` condition vs control | 🔄 Running |
| "Resonance scales favorably" | 1M–50M sweep | ❌ Not yet tested |
