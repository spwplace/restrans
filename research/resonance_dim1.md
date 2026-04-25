# Dimension 1: Related Architectures & Multimodal Transformers

> Research on dual embeddings, disentangled representations, auxiliary embedding spaces in transformers, and orthogonal/neurosymbolic architectures relevant to the Resonance Transformer.

---

## 1. Phase-Coded Memory and Morphological Resonance
**Citation:** Listopad, Aleksandr. "Phase-Coded Memory and Morphological Resonance." arXiv:2511.11848 (2025).  
**URL:** https://arxiv.org/abs/2511.11848

**Summary:** Introduces a wave-based memory representation where embedding vectors are transformed into complex-valued waveforms with amplitude (semantic) and phase (structural) components. Proposes a "resonance score" that generalizes cosine similarity by incorporating phase coherence, enabling better retrieval on negation and compositional queries. Implements this in ResonanceDB.

**Relevance:** This is the closest prior art to the Resonance Transformer's phase-embedding concept. It demonstrates that phase-aware similarity can outperform traditional vector similarity for nuanced semantic tasks. The Resonance Transformer differs by integrating phase into the attention mechanism itself, rather than using it only at retrieval time.

---

## 2. Wave-Based Semantic Memory with Resonance-Based Retrieval
**Citation:** Listopad, Aleksandr. "Wave-Based Semantic Memory with Resonance-Based Retrieval: A Phase-Aware Alternative to Vector Embedding Stores." arXiv:2509.09691 (2025).  
**URL:** https://arxiv.org/abs/2509.09691

**Summary:** Builds on the above by providing detailed empirical evaluation of phase-enriched queries versus cosine-based retrieval. Shows improvements on tasks involving negation, inversion, and contextual shift. Introduces sign-phase mapping for compatibility with standard vector embeddings.

**Relevance:** Provides the foundational evidence that phase-aware representations can capture semantic nuances missed by amplitude-only (standard embedding) approaches. Validates the core hypothesis behind Resonance Transformer's dual embedding strategy.

---

## 3. RoFormer: Enhanced Transformer with Rotary Position Embedding
**Citation:** Su, Jianlin, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864 (2021).  
**URL:** https://arxiv.org/abs/2104.09864

**Summary:** Introduces Rotary Position Embedding (RoPE), which encodes absolute position via a rotation matrix while simultaneously incorporating explicit relative position dependency in self-attention. RoPE has been adopted in LLaMA, Qwen, Gemma, and many modern LLMs. The key insight is that rotating query/key vectors by position-dependent angles naturally induces decaying inter-token dependencies.

**Relevance:** RoPE is the most direct prior art for "phase" concepts in transformers. The rotation matrix used in RoPE is mathematically a phase operation on embedding dimensions. The Resonance Transformer extends this idea from positional encoding to token-level semantic encoding, learning a separate phase embedding space for structural/phonetic information.

---

## 4. An Explainable End-to-End ASR Model with Speech Content-Context Separation (Disentangled-Transformer)
**Citation:** (Authors not fully extracted; paper from arXiv:2411.17846, 2024).  
**URL:** https://arxiv.org/abs/2411.17846

**Summary:** Proposes a Disentangled-Transformer that partitions embeddings into sub-embeddings with different temporal behaviors—content embeddings (rapidly varying) and speaker embeddings (slowly varying). Uses separate Q/K/V projection matrices for each sub-embedding type and adds time-invariant regularization to the speaker branch.

**Relevance:** Demonstrates that splitting a single embedding into multiple semantically distinct sub-embeddings within a transformer is feasible and beneficial. The Resonance Transformer's semantic+phase split follows a similar philosophy, but applies it to language (meaning vs. structure) rather than speech (content vs. speaker).

---

## 5. Adaptive Disentangled Transformer for Sequential Recommendation
**Citation:** (Authors from Tsinghua; paper on sequential recommendation with disentangled transformers).  
**URL:** https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2023_Adaptive%20Disentangled%20Transformer%20for%20Sequential%20Recommendation.pdf

**Summary:** Proposes an adaptive disentangled transformer framework that decomposes user behavior representations into different latent factors. Uses auxiliary independence objectives and reconstruction objectives to ensure disentangled representations remain both independent and information-rich.

**Relevance:** Shows that disentanglement in transformers can be achieved through architectural modifications to the attention mechanism, combined with auxiliary training objectives. The Resonance Transformer could draw on similar techniques to ensure semantic and phase embeddings remain meaningfully separate.

---

## 6. MMFace-DiT: A Dual-Stream Diffusion Transformer
**Citation:** (arXiv:2603.29029, 2026).  
**URL:** https://arxiv.org/abs/2603.29029

**Summary:** Introduces a Dual-Stream Multi-Modal Diffusion Transformer that jointly processes semantic (text) and spatial (masks, sketches) conditions as co-equals in parallel streams, deeply fused via shared RoPE attention at every block. Includes a dynamic Modality Embedder.

**Relevance:** Represents a recent trend toward dual-stream transformer architectures for multimodal tasks. The Resonance Transformer's dual semantic+phase streams share this parallel-processing philosophy, though applied to intra-modal (language-internal) dualities rather than inter-modal ones.

---

## 7. Disentangled Representation Transformer Network for 3D Face Reconstruction
**Citation:** (Springer, The Visual Computer, 2023).  
**URL:** https://link.springer.com/article/10.1007/s00371-023-03202-4

**Summary:** Proposes DRTN, which decomposes face attribute information into identity, expression, and pose branches. Each branch independently regresses one attribute while coupling others, using transformer modules for global information interaction across attributes.

**Relevance:** Demonstrates attribute-specific branching in transformers with cross-attribute coupling. The Resonance Transformer's learnable blend of semantic and phase information can be seen as a soft, learned version of such branching.

---

## 8. Cheap Orthogonal Constraints in Neural Networks
**Citation:** Lezcano-Casado, Mario. "Cheap Orthogonal Constraints in Neural Networks." arXiv:1901.08428 (2019).  
**URL:** https://arxiv.org/abs/1901.08428

**Summary:** Surveys methods for enforcing orthogonality constraints on neural network weights, including manifold optimization, Cayley transforms, and Householder reflections. Shows that orthogonal constraints stabilize optimization and improve generalization.

**Relevance:** The Resonance Transformer's semantic and phase embeddings could potentially benefit from orthogonality constraints to ensure they capture non-redundant information. This paper provides the methodological foundation for such constraints.

---

## 9. Lambda-Orthogonality Regularization for Compatible Representation Learning
**Citation:** Ricci, Simone, et al. "Lambda-Orthogonality Regularization for Compatible Representation Learning." NeurIPS 2025.  
**URL:** https://arxiv.org/abs/2509.16664

**Summary:** Proposes a relaxed orthogonality constraint (lambda-orthogonality) for adapting latent spaces while preserving original learned representations. Balances between affine transformations (adaptable but structure-altering) and orthogonal transformations (structure-preserving but rigid).

**Relevance:** Offers a practical framework for maintaining compatibility between different representation spaces—directly relevant to how the Resonance Transformer might ensure its semantic and phase embeddings remain compatible yet distinct.

---

## 10. Analyzing Transformers in Embedding Space
**Citation:** Dar, Guy, Mor Geva, Ankit Gupta, and Jonathan Berant. "Analyzing Transformers in Embedding Space." arXiv:2209.02535 (2023).  
**URL:** https://arxiv.org/abs/2209.02535

**Summary:** Presents a framework for interpreting all transformer parameters by projecting them into embedding space. Shows that both pretrained and fine-tuned model parameters can be understood as operations in the space of vocabulary items.

**Relevance:** Provides theoretical justification for treating embedding space as the primary interpretable substrate of transformers. The Resonance Transformer's dual embedding spaces (semantic and phase) can both be analyzed through this lens.

---

## Key Findings / Takeaways for Dimension 1

1. **Dual-stream architectures are an emerging pattern** in both multimodal transformers (MMFace-DiT) and single-modal disentanglement (Disentangled-Transformer for ASR, DRTN for faces). The Resonance Transformer fits into this trend by applying dual-stream processing to language's internal structure-meaning duality.

2. **RoPE is the most successful "phase-like" mechanism in transformers**, but it is restricted to positional encoding. The Resonance Transformer generalizes the phase concept to token-level semantic/structural content.

3. **Orthogonality and disentanglement techniques** are well-developed but typically applied to weight matrices or sub-embeddings with explicit regularization. The Resonance Transformer's "learnable blend" offers a softer, data-driven alternative.

4. **Phase-coded memory (ResonanceDB)** validates that phase-aware similarity can improve retrieval, but it treats phase as a post-hoc augmentation to existing embeddings. The Resonance Transformer integrates phase natively into the architecture.

5. **Gap identified:** No prior work combines (a) learned dual embeddings for semantic vs. structural content, (b) phase-based attention bias, and (c) per-dimension learnable blending within a single transformer architecture.
