# Dimension 4: Phase & Frequency in Neural Networks

> Research on phase embeddings, Fourier features, frequency-domain attention, periodic activations, and oscillatory neural dynamics relevant to the Resonance Transformer's phase-based mechanisms.

---

## 1. RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)
**Citation:** Su, Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864 (2021).  
**URL:** https://arxiv.org/abs/2104.09864

**Summary:** Introduces Rotary Position Embedding (RoPE), which encodes absolute position by multiplying context representations with a rotation matrix, simultaneously incorporating explicit relative position dependency in self-attention. RoPE exhibits decaying inter-token dependency with distance and supports linear self-attention variants. Now standard in LLaMA, Qwen, Gemma.

**Relevance:** RoPE is the canonical example of "phase" (rotation angle) being used successfully in transformers. The Resonance Transformer generalizes this idea: instead of encoding only position via rotation, it learns a token-specific phase embedding that captures structural/phonetic properties. The mathematical machinery (rotation matrices, dot-product interaction) is directly analogous.

---

## 2. Fourier Features Let Networks Learn High Frequency Functions
**Citation:** Tancik, Matthew, et al. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." NeurIPS 2020. arXiv:2006.10739.  
**URL:** https://arxiv.org/abs/2006.10739

**Summary:** Shows that mapping input coordinates through random Fourier features (sinusoidal mappings) before passing through an MLP modifies the Neural Tangent Kernel (NTK) eigenvalue spectrum, enabling networks to learn high-frequency functions. Gaussian random Fourier features perform best among tested mappings.

**Relevance:** Provides the theoretical foundation for why sinusoidal/phase-based features help networks capture high-frequency (fine-grained) structure. In language, "high-frequency" structure corresponds to phonetic patterns, rhyme schemes, and morphological fine structure that standard embeddings may miss.

---

## 3. STAF: Sinusoidal Trainable Activation Functions for Implicit Neural Representation
**Citation:** (arXiv:2502.00869, 2025).  
**URL:** https://arxiv.org/abs/2502.00869

**Summary:** Proposes Sinusoidal Trainable Activation Functions (STAF), which generalize periodic activations like SIREN by introducing trainable parameters for frequency and phase. Shows that parameterized Fourier series activations improve performance in high-fidelity signal reconstruction tasks.

**Relevance:** Demonstrates that making periodic/phase components *learnable* (rather than fixed as in standard positional encoding) improves representational capacity. The Resonance Transformer's learnable phase embeddings follow this same principle.

---

## 4. Phase-Coded Memory and Morphological Resonance
**Citation:** Listopad, Aleksandr. "Phase-Coded Memory and Morphological Resonance." arXiv:2511.11848 (2025).  
**URL:** https://arxiv.org/abs/2511.11848

**Summary:** Introduces phase-coded memory where knowledge is stored as amplitude-phase patterns and retrieved via resonance (constructive interference). Demonstrates that phase-aware retrieval outperforms traditional vector similarity on negation, compositional queries, and contextual modulation.

**Relevance:** The closest prior art to the Resonance Transformer's core idea. While ResonanceDB applies phase to memory retrieval, the Resonance Transformer applies phase natively to attention computation. Both share the "resonance" metaphor: tokens "resonate" (attend more strongly) when their phase embeddings align.

---

## 5. A Complex-Valued Oscillatory Neural Network for Storage and Retrieval
**Citation:** Biswas, Dipayan, Sooryakiran Pallikkulath, and V. S. Chakravarthy. "A Complex-Valued Oscillatory Neural Network for Storage and Retrieval of Multidimensional Aperiodic Signals." Frontiers in Computational Neuroscience 15 (2021): 551111.  
**URL:** https://pubmed.ncbi.nlm.nih.gov/34108869/

**Summary:** Shows that networks of complex Hopf oscillators can store multiple time-series patterns via Fourier-like multi-frequency coding, achieving stable phase relationships for memory. Provides a biologically plausible mechanism for phase-coded neural memory.

**Relevance:** Offers biological/neuroscientific grounding for phase-based neural representations. Supports the feasibility of phase-coded mechanisms in artificial networks.

---

## 6. Storage of 7±2 Short-Term Memories in Oscillatory Subcycles
**Citation:** Lisman, John E., and Marco A. Idiart. "Storage of 7±2 Short-Term Memories in Oscillatory Subcycles." Science 267.5203 (1995): 1512-1515.  
**URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3648857/

**Summary:** Proposes that items in working memory are maintained in distinct gamma oscillation phases within a theta cycle. This theta-gamma neural code is a foundational biological precedent for phase-based information indexing.

**Relevance:** Provides strong biological motivation for the Resonance Transformer's phase mechanism. The idea that different "items" (here, different structural patterns) can be indexed by phase is directly analogous to the phase embedding concept.

---

## 7. Holographic Reduced Representations (HRR)
**Citation:** Plate, Tony A. "Holographic Reduced Representations." IEEE Transactions on Neural Networks 6.3 (1995): 623-641.  
**URL:** https://cogsci.ucsd.edu/~sereno/170/readings/06-Holographic.pdf

**Summary:** Presents HRR, a distributed representation scheme using circular convolution to bind vectors and circular correlation to decode them. Shows that superposed memories can be stored and retrieved via Fourier-like operations.

**Relevance:** Holographic representations use phase-like operations (circular convolution in the Fourier domain) for associative memory. The Resonance Transformer's cosine-similarity phase bias in attention is mathematically related to correlation operations in HRR.

---

## 8. Frequency-Domain Fusion Transformer for Image Inpainting
**Citation:** (arXiv:2506.18437, 2025).  
**URL:** https://arxiv.org/abs/2506.18437

**Summary:** Proposes a transformer that combines wavelet transform and Gabor filtering in attention to enhance multi-scale structural modeling. Uses a learnable frequency-domain filter based on FFT to replace the feedforward network, enabling adaptive noise suppression and detail retention.

**Relevance:** Shows that frequency-domain operations can be integrated directly into transformer blocks (not just as preprocessing). The Resonance Transformer's phase attention bias operates similarly—modulating attention in a "frequency-like" (phase alignment) domain.

---

## 9. A Modified Transformer based on Adaptive Frequency Enhanced Attention
**Citation:** (Nature Scientific Reports, 2025).  
**URL:** https://www.nature.com/articles/s41598-025-18187-4

**Summary:** Proposes ALMFormer with adaptive frequency-domain attention based on Discrete Cosine Transform (DCT). Integrates frequency-domain interaction, low-frequency enhancement, and local self-attention. Achieves O(BCL log P) complexity vs. O(BCL^2) for standard attention.

**Relevance:** Frequency-domain attention is computationally efficient and captures periodic patterns well. The Resonance Transformer's phase-based attention achieves similar computational complexity benefits while targeting language-specific periodicity (rhyme, meter, morphology).

---

## 10. SIREN: Implicit Neural Representations with Periodic Activation Functions
**Citation:** Sitzmann, Vincent, et al. "Implicit Neural Representations with Periodic Activation Functions." NeurIPS 2020.  
**URL:** https://www.vincentsitzmann.com/siren/

**Summary:** Uses sinusoidal activation functions (instead of ReLU) for implicit neural representations of signals. Networks with periodic activations can represent fine details and derivatives accurately, serving as neural signal representations.

**Relevance:** SIREN demonstrates that periodic/oscillatory components are essential for capturing fine structure in continuous signals. The Resonance Transformer applies the same insight to discrete language sequences via phase embeddings.

---

## Key Findings / Takeaways for Dimension 4

1. **Phase and rotation are already successful in transformers** via RoPE, but restricted to positional encoding. The Resonance Transformer is the first to generalize phase to token-level content embeddings.

2. **Fourier features and periodic activations** enable networks to learn high-frequency structure that standard architectures miss. In language, this corresponds to phonetic, morphological, and syntactic fine structure.

3. **Biological precedents** (theta-gamma coding, complex oscillatory networks) strongly support the viability of phase-based information indexing in neural systems.

4. **Frequency-domain attention mechanisms** in vision and signal processing transformers show that attention can be productively computed in non-spatial domains. The Resonance Transformer's phase-biased attention is a linguistic analog.

5. **Holographic reduced representations** demonstrate that distributed memory systems can use phase-correlation operations (Fourier-domain) for associative retrieval, prefiguring the resonance attention mechanism.

6. **Gap identified:** No prior transformer architecture learns token-level phase embeddings that capture structural/phonetic content and uses phase coherence as an attention bias. RoPE uses fixed position-dependent phases; ResonanceDB uses phase at retrieval time; SIREN uses periodic activations in MLPs. The Resonance Transformer combines all three ideas in a unified architecture.
