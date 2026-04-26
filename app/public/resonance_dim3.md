# Dimension 3: Model Compression & Representation Analysis

> Research on transformer quantization, embedding compression, effective rank, mutual information in layers, and the geometric structure of learned representations.

---

## 1. A Survey on Transformer Compression
**Citation:** Zhu, Yunshan, et al. "A Survey on Transformer Compression." arXiv:2402.05964 (2024).  
**URL:** https://arxiv.org/abs/2402.05964

**Summary:** Comprehensive survey covering quantization (PTQ and QAT), pruning, knowledge distillation, efficient architecture design, and tensor decomposition for transformers. Discusses unique challenges of compressing attention and FFN modules. Covers methods including GPTQ, AWQ, SmoothQuant, LLM.int8(), and low-rank factorization approaches.

**Relevance:** Provides the broad landscape for the Resonance Transformer's compression studies. The dual-stream architecture introduces new compression opportunities (e.g., compressing phase and semantic streams differently) not addressed by existing surveys.

---

## 2. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
**Citation:** Frantar, Elias, et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323 (2022).  
**URL:** https://arxiv.org/abs/2210.17323

**Summary:** Proposes a one-shot weight quantization method based on approximate second-order information. Can quantize 175B-parameter models to 3-4 bits with negligible accuracy degradation. Enables running large models on a single GPU.

**Relevance:** The Resonance Transformer's compression analysis can benchmark against GPTQ. A key research question is whether dual-stream representations are more or less robust to quantization than standard single-stream embeddings.

---

## 3. Online Embedding Compression for Text Classification using Low Rank Matrix Factorization
**Citation:** Acharya, Anish, et al. "Online Embedding Compression for Text Classification using Low Rank Matrix Factorization." AAAI 2019.  
**URL:** https://arxiv.org/abs/1811.00641

**Summary:** Proposes compressing word embedding layers via SVD-based low-rank factorization during training. Achieves up to 90% model size reduction with minimal accuracy impact. Introduces Cyclically Annealed Learning Rate (CALR) schedule.

**Relevance:** Embedding matrices are the size bottleneck in many NLP models. The Resonance Transformer has *two* embedding matrices, making compression especially important. This paper shows that low-rank structure exists in embeddings and can be exploited.

---

## 4. TensorGPT: Efficient Compression of the Embedding Layer in LLMs based on Tensor-Train Decomposition
**Citation:** Xu, Mingxue, Yao Lei Xu, and Danilo P. Mandic. "TensorGPT: Efficient Compression of Large Language Models based on Tensor-Train Decomposition." arXiv:2307.00526 (2023).  
**URL:** https://arxiv.org/abs/2307.00526

**Summary:** Compresses embedding layers using Tensor-Train Decomposition (TTD), treating each token embedding as a Matrix Product State (MPS). Achieves 38x compression for GPT-2 embedding layers, with some settings actually improving performance (likely due to over-parameterization in original).

**Relevance:** Shows that embedding layers have massive redundancy. The Resonance Transformer's dual embeddings (semantic + phase) each present compression opportunities. The tensor-train approach could be applied independently to each stream.

---

## 5. A Survey on Model Compression for Large Language Models (TACL)
**Citation:** (TACL 2024).  
**URL:** https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00704/125482

**Summary:** Reviews quantization, pruning, knowledge distillation, and low-rank factorization for LLMs. Covers methods like LPLR (randomized low-rank + low-precision), ASVD (activation-aware SVD), and LASER (layer-selective rank reduction).

**Relevance:** LASER is particularly relevant: it shows that selectively reducing higher-order components of weight matrices can improve model performance on rare data. The Resonance Transformer's phase stream may serve a similar "higher-order" role that benefits from selective rank reduction.

---

## 6. MiniLLM: On-Policy Distillation of Large Language Models
**Citation:** Gu, Yuxian, et al. "MiniLLM: On-Policy Distillation of Large Language Models." arXiv:2306.08543 (2023).  
**URL:** https://arxiv.org/abs/2306.08543

**Summary:** Proposes reverse-KL divergence for knowledge distillation of generative LLMs. Student models learn to focus on high-probability outcomes rather than matching the full teacher distribution. Scales to models from 120M to 13B parameters.

**Relevance:** Knowledge distillation is a key compression strategy. The Resonance Transformer's dual-stream architecture raises interesting questions: can one stream be distilled more aggressively than the other? Does the phase stream distill more cleanly than the semantic stream?

---

## 7. Entropy and Mutual Information in Models of Deep Neural Networks
**Citation:** (NeurIPS 2019).  
**URL:** https://proceedings.neurips.cc/paper/7453-entropy-and-mutual-information-in-models-of-deep-neural-networks.pdf

**Summary:** Presents a tractable method to compute entropy and mutual information between layers in deep networks. Shows that compression can happen during learning even with ReLU activations. Uses replica method from statistical physics.

**Relevance:** Provides theoretical tools for analyzing information flow in the Resonance Transformer. Can help answer: How much information does the phase stream add beyond the semantic stream? Is there redundancy between the two streams?

---

## 8. On Representation Rank Collapse in Deep Neural Networks
**Citation:** Siddiqui, Mohammed Faisal Shahzad. "On Representation Rank Collapse in Deep Neural Networks." IRJET (2024).  
**URL:** https://www.irjet.net/archives/V13/i3/IRJET-V13I03166.pdf

**Summary:** Analyzes representation rank collapse—the tendency of activation matrices to become low-rank during training. Proposes a rank-preserving regularization based on log-determinant of the activation Gram matrix to encourage diverse feature representations.

**Relevance:** The Resonance Transformer's dual embeddings are intended to increase effective rank by encoding complementary information in two streams. This paper provides metrics (activation matrix rank, Gram matrix log-determinant) for measuring whether this goal is achieved.

---

## 9. Why Do Neural Networks Forget: A Study of Collapse in Continual Learning
**Citation:** Zhu, Yunqin, et al. "Why Do Neural Networks Forget: A Study of Collapse in Continual Learning." arXiv:2603.04580 (2026).  
**URL:** https://arxiv.org/abs/2603.04580

**Summary:** Investigates the link between catastrophic forgetting and representation collapse (measured via effective rank, eRank). Shows that forgetting and collapse are strongly correlated across MLP, ConvGRU, ResNet-18, and Bi-ConvGRU architectures.

**Relevance:** If the Resonance Transformer's dual embeddings increase effective rank, they may also improve resistance to forgetting. This paper's eRank metric provides a tool for testing this hypothesis.

---

## 10. Neural Collapse is Globally Optimal in Deep Regularized ResNets and Transformers
**Citation:** (NeurIPS 2025).  
**URL:** https://neurips.cc/virtual/2025/poster/119611

**Summary:** Shows that neural collapse (where class features collapse to simplex equilibria and classifiers align with features) is the global optimum in deep regularized networks including transformers. Provides theoretical characterization of the geometric structure learned at convergence.

**Relevance:** Neural collapse describes the geometry of final-layer representations. The Resonance Transformer's phase embeddings may exhibit different geometric structure (e.g., toroidal or periodic organization) compared to standard semantic embeddings, offering a new lens on representation geometry.

---

## Key Findings / Takeaways for Dimension 3

1. **Embedding layers are highly compressible**—up to 38x compression with tensor methods, or 90% with low-rank factorization, often with minimal or no accuracy loss. The Resonance Transformer's two embedding matrices present both a challenge (more parameters) and an opportunity (two independent compression targets).

2. **Rank collapse is a real phenomenon** in deep networks, limiting effective capacity. Dual-stream architectures that enforce complementary information in separate streams may mitigate collapse and increase effective rank.

3. **Layer-selective rank reduction (LASER)** shows that different layers benefit from different compression strategies. The Resonance Transformer could exploit this by applying different compression rates to semantic vs. phase streams.

4. **Mutual information between layers** is a principled framework for analyzing what different network components encode. This can validate that the phase stream captures genuinely non-redundant information.

5. **Perplexity stability under perturbation** (a Resonance Transformer metric) connects to the broader literature on quantization robustness. GPTQ and AWQ show that weight perturbation robustness varies significantly across architectures and layers.

6. **Gap identified:** No prior work systematically studies how dual-stream representations compress relative to single-stream ones, or whether structural/phase information is inherently more or less robust to quantization than semantic information.
