# Resonance Transformers: Dual-Stream Architectures with Phase-Structured Attention

## Abstract
### Research Summary
#### Language models represent tokens solely by semantic content, ignoring the structural regularities that govern linguistic organization
#### We propose Resonance Transformers, which learn a secondary phase embedding alongside standard semantic embeddings
#### Resonance-biased attention allows structurally similar tokens to attend to each other more readily, while a learnable blend controls their interpolation
#### On synthetic structured data, Resonance models exhibit comparable scaling to standard transformers, with phase streams showing dramatically higher compressibility
#### The phase stream tolerates aggressive quantization (to 2-bit) without performance degradation, whereas semantic quantization causes significant perplexity increases

## 1. Introduction (~1500 words)
### 1.1 Background and Motivation
#### 1.1.1 Transformers map tokens to a single semantic latent space, collapsing orthographic, phonetic, and syntactic structure into one representation
#### 1.1.2 Human language processing engages multiple modalities simultaneously: meaning, sound, rhythm, and grammatical role
#### 1.1.3 Prior work on multimodal transformers addresses cross-modal inputs (vision + text) but not intra-modal structure within language itself
### 1.2 Research Gap and Problem Statement
#### 1.2.1 No existing architecture learns dual semantic/structural representations natively within the transformer attention mechanism
#### 1.2.2 Phase-coded retrieval systems exist but have not been integrated into end-to-end language model training
#### 1.2.3 The hypothesis: structural regularities provide a compressed inductive bias that can compensate for parameter scarcity or accelerate learning
### 1.3 Contribution Statement
#### 1.3.1 Architecture contribution: Resonance Transformer with token-level phase embeddings, resonance-biased attention, and per-dimension learnable blending
#### 1.3.2 Empirical contribution: systematic analysis of scaling, compressibility, and representation geometry showing phase streams are ~8x more compressible than semantic streams
#### 1.3.3 Training contribution: proof-walk synthetic data framework from lambda calculus for instilling structural reasoning
#### 1.3.4 Robustness contribution: perplexity stability analysis under perturbation revealing differential sensitivity of semantic and phase parameters
### 1.4 Paper Organization
#### 1.4.1 Section 2 reviews related work on multimodal transformers, phase mechanisms, and synthetic training data
#### 1.4.2 Section 3 presents the Resonance Transformer architecture and the lambda calculus proof-walk training framework
#### 1.4.3 Section 4 reports experiments on scaling laws, compressibility, gestalt stream analysis, and perturbation stability
#### 1.4.4 Section 5 discusses implications and limitations; Section 6 concludes

## 2. Related Work (~1800 words, 1 comparison table)
### 2.1 Multimodal and Disentangled Transformer Architectures
#### 2.1.1 Disentangled transformers for speech separate content and speaker representations with hard regularization
#### 2.1.2 Vision-language models (CLIP, DALL-E) fuse cross-modal embeddings but do not address intra-modal structure
#### 2.1.3 RoPE encodes position via rotation matrices — a fixed phase mechanism restricted to positional information
### 2.2 Phase and Frequency Mechanisms in Neural Networks
#### 2.2.1 Phase-coded memory architectures for retrieval use phase similarity for external knowledge compression
#### 2.2.2 Fourier features and periodic activations (SIREN, STAF) enable high-frequency signal learning
#### 2.2.3 Biological theta-gamma codes suggest phase-locking as an indexing mechanism for working memory
### 2.3 Synthetic Data and Formal Language Pretraining
#### 2.3.1 S4Eq and Neural Lambda Calculus demonstrate that transformers can learn formal reasoning from synthetic programs
#### 2.3.2 ContraCode uses compiler transformations as contrastive augmentations for code representation learning
#### 2.3.3 Active synthetic data generation with multi-agent quality control emerges as a best practice
### 2.4 Model Compression and Representation Geometry
#### 2.4.1 Embedding layers are massively redundant: TensorGPT achieves 38x compression; low-rank factorization reaches 90% sparsity
#### 2.4.2 Rank collapse and neural collapse describe the geometric structure of converged representations
#### 2.4.3 Layer-selective rank reduction (LASER) shows different layers benefit from different compression profiles
### 2.5 Positioning Against Prior Work
#### 2.5.1 Comparison table: Resonance Transformer vs RoPE, Phase-Coded Memory, Disentangled-Transformer, ContraCode, TensorGPT on 6 dimensions (architecture, training integration, attention bias, learnable blending, compression, synthetic data)
#### 2.5.2 Key distinction: Resonance Transformers learn token-content phase (not position-dependent) and integrate phase bias directly into causal self-attention during training

## 3. Methodology (~2500 words, 2 figures, 2 algorithms)
### 3.1 Preliminaries: Standard Causal Transformer
#### 3.1.1 Notation: vocabulary V, sequence length S, embedding dimension D, token indices x ∈ V^S
#### 3.1.2 Standard architecture: token embedding E ∈ R^{V×D}, positional encoding P ∈ R^{S×D}, L transformer blocks, causal attention mask
#### 3.1.3 Loss: cross-entropy between logits and next-token targets
### 3.2 Resonance Transformer Architecture
#### 3.2.1 Dual embeddings: semantic E_s ∈ R^{V×D} and phase E_p ∈ R^{V×F} where F=32 frequency dimensions
#### 3.2.2 Phase projection: h_phase = W_p · E_p(x) where W_p ∈ R^{F×D}
#### 3.2.3 Learnable blend: α ∈ R^D with sigmoid gating; blended embedding h = σ(α) ⊙ h_sem + (1 − σ(α)) ⊙ h_phase
#### 3.2.4 Resonance matrix: R[i,j] = (1/F) Σ_f cos(E_p[x_i, f] − E_p[x_j, f])
#### 3.2.5 Resonance-biased attention: attn = softmax( (QK^T)/√d_head + R · w_r ) where w_r is a learnable scalar weight
#### 3.2.6 Phonetic initialization: rhyme groups (from CMUdict) share similar phase vectors, providing structural prior
### 3.3 Synthetic Data Generation: Lambda Calculus Proof Walks
#### 3.3.1 Generate well-typed simply-typed lambda terms with de Bruijn indices
#### 3.3.2 Proof-walk dataset: for each type judgement Γ ⊢ ? : T, generate k distinct programs/proofs satisfying it
#### 3.3.3 Contrastive group training: pull programs within the same proof group together, push different groups apart via InfoNCE loss
#### 3.3.4 Semantic-preserving mutation engine: β/η expansion and reduction, let-binding introduction, dead code manipulation — all verified by normal-form bisimilarity
#### 3.3.5 Scheduled mutation curriculum: number of accumulated mutations increases from 0 to 10,000 over training
### 3.4 Training Procedure
#### 3.4.1 Optimizer: AdamW with cosine annealing, weight decay 0.01, gradient clipping at 1.0
#### 3.4.2 Hyperparameters: batch size 32, learning rate 3e-4, dropout 0.1
#### 3.4.3 Evaluation metrics: validation perplexity, top-1 accuracy, compression ratio vs. perplexity trade-off

## 4. Experiments and Results (~2500 words, 4 figures, 3 tables)
### 4.1 Experimental Setup
#### 4.1.1 Dataset: structured synthetic tokens with 5,000 vocabulary, 50,000 training sequences, 10,000 validation sequences
#### 4.1.2 Local Markov dependencies (80% continuity) and long-range correlations (every 8th token)
#### 4.1.3 Model scales: Small (1.4M params), Medium (5.5M params), and scaling grid (1x, 2x, 4x)
#### 4.1.4 Baselines: Standard Transformer matched for parameter count and training regime
### 4.2 Scaling Behavior
#### 4.2.1 Training curves show stable convergence for both architectures
#### 4.2.2 Scaling law plot: log(params) vs. log(perplexity) for Standard and Resonance variants
#### 4.2.3 Resonance adds ~3% parameter overhead (phase embeddings + projection); training time overhead ~5%
### 4.3 Compressibility of Learned Parameters
#### 4.3.1 Uniform quantization: 8-bit, 4-bit, 2-bit bucket rounding
#### 4.3.2 Magnitude pruning at 30%, 50%, 70% sparsity
#### 4.3.3 Table: compression ratio vs. perplexity for Standard and Resonance
#### 4.3.4 Key result: aggressive phase-only quantization (2-bit phase + 8-bit semantic) matches or slightly improves baseline perplexity, while semantic-only quantization degrades performance by +38.5 perplexity
### 4.4 Gestalt Stream Analysis
#### 4.4.1 PCA dimensionality: semantic stream requires 234 dimensions for 95% variance; phase stream requires only 29 (projected) or 31 (raw)
#### 4.4.2 Cross-predictability: R² ≈ 0 for both directions, confirming orthogonal information content
#### 4.4.3 Resonance matrix effective rank ≈ 1.0, indicating a single dominant structure
#### 4.4.4 Alpha blend distribution concentrates near 0.5 (mean 0.5009, std 0.0005)
### 4.5 Perturbation Stability
#### 4.5.1 Perturbation types: phase embeddings, semantic embeddings, full weights, blend parameter, resonance weight
#### 4.5.2 Gaussian noise at scales σ ∈ {0.001, 0.01, 0.05, 0.1, 0.2, 0.5}
#### 4.5.3 Result: phase embeddings and resonance weight show near-perfect stability (ratio ≈ 1.0 across all σ); semantic embeddings and full weights degrade monotonically
### 4.6 Ablation and Sensitivity
#### 4.6.1 Random vs. phonetic phase initialization: minimal difference on synthetic data (expected: no rhyme structure in tokens)
#### 4.6.2 Resonance weight ablation: setting w_r = 0 recovers standard attention with phase embeddings still blended

## 5. Discussion (~1200 words)
### 5.1 Interpretation of Results
#### 5.1.1 Phase stream as a compressed structural index: low effective dimensionality and high quantization tolerance suggest phase acts as a coarse-grained similarity registry
#### 5.1.2 Orthogonality between streams: near-zero cross-predictability implies the model has learned to factor information, not duplicate it
#### 5.1.3 Stability asymmetry: semantic parameters carry task-critical information; phase parameters are either redundant or provide robust regularization
### 5.2 Limitations
#### 5.2.1 Synthetic data lacks true linguistic structure; phonetic initialization has no effect on random tokens
#### 5.2.2 Models are undertrained relative to Chinchilla-optimal (2-5 epochs vs. hundreds)
#### 5.2.3 Scale study only reaches 5.7M parameters; behavior at 100M+ parameters is unknown
#### 5.2.4 Single-run results without variance estimates; stochasticity not quantified
### 5.3 Implications and Future Directions
#### 5.3.1 Edge deployment: phase stream can be aggressively compressed for on-device models
#### 5.3.2 Synthetic pretraining: lambda calculus proof walks as a structural reasoning curriculum before natural language finetuning
#### 5.3.3 RL fine-tuning: resonance attention may provide a natural inductive bias for multi-step reasoning
#### 5.3.4 Scaling to 100x: framework is ready; requires GPU cluster and natural language data

## 6. Conclusion (~500 words)
### 6.1 Summary of Contributions
#### 6.1.1 We introduced Resonance Transformers, a dual-stream architecture integrating learnable phase embeddings into causal self-attention
#### 6.1.2 Empirical analysis reveals phase streams are ~8x more compressible than semantic streams and remarkably stable under perturbation
#### 6.1.3 We released a complete training framework including lambda calculus synthetic data generation and contrastive proof-walk training
### 6.2 Future Work
#### 6.2.1 Large-scale pretraining on natural language (OpenWebText, C4) to validate phonetic initialization and scaling laws
#### 6.2.2 Reinforcement learning fine-tuning with resonance-biased reward shaping
#### 6.2.3 Extension to other modalities: code syntax trees, molecular graphs, musical scores

# References
## resonance_outline_references_raw.md
- **Type**: Citation collection
- **Description**: All sources from outline-stage literature scanning
- **Path**: /mnt/agents/output/research/resonance_dim1.md, resonance_dim2.md, resonance_dim3.md, resonance_dim4.md, resonance_cross_verification.md, resonance_insight.md
