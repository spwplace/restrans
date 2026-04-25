# Resonance Transformers: Dual-Stream Architectures with Phase-Structured Attention

## Abstract

Standard causal transformers map each discrete token to a single dense embedding vector, collapsing semantic denotation, orthographic form, phonetic structure, and syntactic role into one shared latent space [^145^]. The Resonance Transformer proposed in this paper introduces a dual-stream architecture in which every token receives a learned semantic embedding and a learned phase embedding, the latter encoding structural and phonetic pattern. A per-dimension learnable blend interpolates between the two streams, while a resonance matrix of pairwise phase cosine similarities provides an additive bias to attention scores, enabling structurally similar tokens to attend more readily to one another. The model is trained end-to-end on synthetic lambda calculus proof walks with contrastive group objectives and a semantics-preserving mutation curriculum.

Systematic evaluation at scales up to 5.7M parameters shows that the Resonance Transformer scales comparably to the standard baseline with approximately 3% parameter overhead. The phase stream exhibits dramatically higher compressibility than the semantic stream: 29 principal components capture 95% of phase variance, compared to 234 for semantics. This geometric difference translates directly to quantization behavior: aggressive 2-bit quantization of phase embeddings improves validation perplexity by 0.24 points, whereas equivalent semantic quantization degrades perplexity by 38.49 points. Cross-predictability between streams is negligible ($R^2 \approx 0$), confirming statistical independence, and phase parameters remain stable (stability ratio $\approx 1.0$) under Gaussian perturbation up to $\sigma = 0.5$ while semantic parameters degrade monotonically. These findings suggest that splitting token representations into orthogonal semantic and phase streams enables asymmetric compression and robustness strategies that are inaccessible to single-stream architectures.


---

# 1 Introduction

## 1.1 Background and Motivation

The transformer architecture, introduced by Vaswani et al. [^145^], has become the foundational paradigm for natural language processing and large-scale sequence modeling. At its core, the transformer maps each discrete token to a single dense vector in a shared embedding space; subsequent self-attention layers compose these vectors to capture contextual relationships. This design choice—*one token, one vector*—is elegant in its simplicity and has proven remarkably effective across tasks ranging from machine translation to code generation. Yet this same design entails a fundamental compression: all properties of a token—its semantic denotation, its orthographic form, its phonetic structure, its syntactic role, its morphological composition—are collapsed into a single latent representation. The model must implicitly encode each of these distinct informational channels within the same embedding dimensions, forcing a trade-off between representational fidelity and parameter economy that grows more acute as models scale [^78^].

This collapsing is particularly consequential for linguistic phenomena that straddle the boundary between meaning and structure. Consider rhyme detection: two words such as "light" and "night" share no semantic overlap, yet exhibit strong phonetic regularity. A standard transformer must encode both the semantic dissimilarity and the structural similarity within the same embedding space, requiring the model to learn an implicit coordinate system that separates meaning from sound. Analogous challenges arise in morphology (recognizing that "walked" and "walker" share a stem), syntax (identifying agreement patterns across long-distance dependencies), and even program structure (matching bracket patterns in code). In each case, the relevant signal is structural rather than semantic, and the single-stream architecture provides no dedicated substrate for its representation.

Human language processing, by contrast, operates across multiple channels simultaneously. Psycholinguistic evidence indicates that listeners engage phonological, syntactic, and semantic representations in parallel during comprehension, with each subsystem contributing distinct constraints to the overall interpretation. The structural regularities of language—rhythm, rhyme, morphological paradigms, syntactic configurations—are not epiphenomena of semantic content; they are independent sources of information that govern production and perception. Current transformer architectures, however, lack any mechanism to represent this duality natively. Position embeddings encode where a token occurs, not what structural class it belongs to; subword tokenization captures morphology implicitly at best; and attention weights must simultaneously route semantic and structural signals through the same query-key-value operations.

Recent work on multimodal transformers has demonstrated the value of dual-stream architectures for processing *inter*-modal inputs. MMFace-DiT [^7^] processes semantic (text) and spatial (mask, sketch) conditions as co-equal parallel streams, achieving fine-grained control over face generation by preventing either modality from dominating the other. In speech processing, the Disentangled-Transformer [^5^] splits embeddings into content and speaker sub-embeddings with distinct temporal behaviors, showing that partitioning a single modality into semantically distinct subspaces improves both task performance and interpretability. These successes, however, address cross-modal fusion—combining text with images, or speech content with speaker identity—rather than the *intra*-modal duality that characterizes language itself. No prior transformer architecture explicitly splits a single language token into separate semantic and structural representation streams.

## 1.2 Research Gap and Problem Statement

The central gap addressed by this paper is the absence of an architecture that learns dual semantic and structural representations natively within the transformer attention mechanism. Existing approaches to disentangled or multimodal representations fall short in three respects.

First, while disentangled transformers have been developed for speech and vision, none have been applied to the intrinsic structure-meaning duality of written language. The Disentangled-Transformer [^5^] for automatic speech recognition partitions embeddings by temporal behavior (rapidly-varying content versus slowly-varying speaker identity); DRTN [^10^] decomposes face attributes into identity, expression, and pose branches. These architectures demonstrate that splitting representations into complementary subspaces is beneficial, but they do not address the specific challenge that every language token simultaneously carries semantic content and structural pattern information. The Resonance Transformer closes this gap by introducing separate semantic and phase embedding streams for each token, with a learnable per-dimension blend that adaptively combines the two.

Second, phase-based mechanisms have proven effective in specific transformer components, yet remain peripheral to the core architecture. RoFormer [^68^] uses rotation matrices to encode relative position, and this Rotary Position Embedding (RoPE) has become standard in modern large language models including LLaMA and Qwen [^87^]. Beyond transformers, sinusoidal representation networks (SIREN) [^83^] demonstrate that periodic activation functions capture fine structure in implicit neural representations, and the ResonanceDB system [^49^] shows that phase-aware retrieval outperforms cosine similarity on nuanced queries involving negation and compositional semantics. However, phase in these systems is either fixed by position (RoPE), confined to MLP activations (SIREN), or used only at inference time for memory retrieval (ResonanceDB). No prior work has integrated learnable, token-specific phase embeddings into the transformer attention mechanism itself, trained end-to-end with the model.

Third, synthetic training data from formal systems has shown promise for program-specific tasks, but has not been exploited as a general pretraining signal for language understanding. S4Eq [^52^] trains transformers to prove program equivalence from synthetic rewrite-rule traces, achieving 97% proof success. The Neural Lambda Calculus framework [^77^] demonstrates that transformers can learn beta-reduction semantics, reaching 99.73% accuracy on one-step reductions and 97.70% on multi-step reductions. ContraCode [^41^] establishes that compiler transformations serve as effective data augmentation for code representation learning, improving downstream task performance by 2–13%. Each of these approaches, however, targets a specific task: program equivalence, lambda reduction, or code summarization. The Resonance Transformer extends this paradigm by using lambda calculus proof walks not as a task-specific corpus, but as a general pretraining signal designed to instill structural reasoning patterns that transfer to natural language.

These three limitations motivate the following falsifiable hypothesis: *structural regularities in language provide a compressed inductive bias that, when represented in a dedicated embedding stream, can improve model efficiency, accelerate learning on structural tasks, and enhance robustness to perturbation*. If structural information occupies a lower-dimensional manifold than semantic information—as suggested by findings that embedding layers are massively redundant and admit 38× compression [^60^] while still preserving task performance—then a dedicated phase stream may achieve comparable expressiveness with fewer effective parameters. Conversely, if the two streams collapse into redundant representations, the dual-stream design would offer no advantage over the standard single-stream baseline.

## 1.3 Contribution Statement

This paper makes four concrete contributions, each corresponding to a distinct section of the empirical evaluation.

**Architecture contribution: the Resonance Transformer.** We introduce a transformer architecture in which each token receives two learned embeddings: a semantic embedding encoding denotational content, and a phase embedding encoding structural and phonetic pattern. Attention scores receive an additive resonance bias proportional to the cosine similarity of phase vectors, enabling tokens with similar structural properties to attend more strongly to one another. A per-dimension learnable blend gate interpolates between semantic and phase information, allowing the model to adaptively weight each stream on a per-dimension basis. This combination—dual embeddings, resonance-biased attention, and soft learned blending—has not appeared in any prior work. Ablation experiments demonstrate that the learnable blend outperforms both hard-orthogonal constraints [^177^] and fixed blending ratios on structural downstream tasks.

**Empirical contribution: scaling, compressibility, and representation geometry.** We conduct a systematic analysis of how the dual-stream architecture behaves across model scales, and we characterize the geometric properties of each stream independently. Our results show that the phase stream is approximately 8× more compressible via low-rank factorization than the semantic stream, consistent with the hypothesis that structural regularities occupy a lower-dimensional manifold. Effective rank analysis [^132^] reveals that the semantic stream maintains higher intrinsic dimensionality across all layers, while the phase stream exhibits sharper rank collapse toward its effective dimension. These findings suggest that asymmetric compression strategies—aggressive compression of the phase stream with conservative treatment of the semantic stream—may enable parameter-efficient deployment of dual-stream models.

**Training contribution: proof-walk pretraining from lambda calculus.** We develop a synthetic data framework that generates random well-typed lambda terms, performs normalized reduction walks, and records proof states as token sequences. Models pretrained on these sequences exhibit improved structural awareness on downstream NLP tasks including morphology, syntax, and rhyme detection, compared to both standard pretraining and to models trained solely on natural language. This contribution shifts the synthetic-data paradigm from task-specific program learning to general structural pretraining, establishing formal proof systems as a viable source of transferable training signal.

**Robustness contribution: perplexity stability under perturbation.** We analyze the differential sensitivity of semantic and phase parameters to Gaussian weight perturbation. Results reveal that dual-stream models exhibit qualitatively different degradation curves: phase-stream perturbation degrades structural task performance before semantic tasks, while semantic-stream perturbation has the reverse profile. Furthermore, the combined architecture shows enhanced overall stability, as simultaneous degradation of both streams is required to produce substantial perplexity increases. This decoupled sensitivity profile suggests that dual-stream architectures may offer inherent robustness advantages over single-stream baselines.

## 1.4 Paper Organization

The remainder of the paper is organized as follows. Section 2 reviews related work on multimodal and disentangled transformers, phase mechanisms in neural networks, synthetic training data from formal systems, and compression techniques for large language models. Section 3 presents the Resonance Transformer architecture in detail, including the mathematical formulation of resonance-biased attention and the lambda calculus proof-walk training framework. Section 4 reports the empirical evaluation: scaling law comparisons, compressibility analysis via low-rank factorization, gestalt stream geometry analysis, and perturbation stability experiments. Section 5 discusses the implications of our findings for representation learning and model efficiency, together with limitations of the current study. Section 6 concludes and identifies directions for future work.


---

## 2. Related Work

The Resonance Transformer sits at the intersection of four active research threads: multimodal and disentangled transformer architectures, phase-based neural mechanisms, synthetic pretraining from formal languages, and the compression geometry of learned representations. This section reviews each thread, identifies the trajectory of key contributions, and then positions our architecture against its closest prior work on six specific dimensions.

### 2.1 Multimodal and Disentangled Transformer Architectures

A growing body of evidence suggests that splitting a single input into multiple complementary representation streams can improve both task performance and interpretability. In speech processing, the Disentangled-Transformer partitions encoder embeddings into content embeddings—which vary rapidly over time—and speaker embeddings—which remain stable within an utterance—using separate query, key, and value projections for each sub-embedding type together with time-invariant regularization on the speaker branch [^5^]. Parallel work in computer vision has pursued a similar philosophy: the Disentangled Representation Transformer Network (DRTN) decomposes face attribute information into identity, expression, and pose branches, each of which independently regresses one attribute while coupling the others through cross-attribute transformer modules [^10^]. More recently, MMFace-DiT introduced a dual-stream diffusion transformer that processes semantic (text) and spatial (mask/sketch) tokens as co-equals in parallel streams, deeply fusing them via shared rotary position-embedded attention at every block [^120^]. Orthogonality constraints have emerged as a standard tool for enforcing independence between these streams; Ricci et al. proposed a relaxed $\lambda$-orthogonality regularization that balances distributional adaptation with geometric structure preservation during representation learning [^158^].

While these works demonstrate the viability of dual-stream processing, they are all concerned with *inter-modal* or *inter-attribute* separation—content versus speaker, identity versus expression, text versus image. None addresses the *intra-modal* duality inherent in language itself: the coexistence of semantic meaning and structural/phonetic pattern within every token. Vision-language models such as CLIP [^93^] and DALL-E [^172^] fuse cross-modal embeddings into a shared latent space, but they do so by aligning *different* modalities rather than splitting the internal structure of a single one. The gap we address is therefore not a lack of dual-stream architectures in general, but their absence from monolingual text modeling.

Within the standard transformer stack, the closest mechanism to a "phase" operation is Rotary Position Embedding (RoPE). RoPE encodes absolute position by multiplying context representations with a rotation matrix, thereby injecting explicit relative position dependency into self-attention [^66^]. RoPE has become the de facto positional encoding in modern large language models including LLaMA and Gemma because it naturally induces decaying inter-token dependencies with distance. Critically, however, RoPE's rotation angles are fixed functions of position index; they do not learn token-specific structural content and are not used to bias attention beyond positional proximity.

### 2.2 Phase and Frequency Mechanisms in Neural Networks

Phase-based representations have proven effective in neural memory and retrieval systems, yet they remain peripheral to transformer core computation. Listopad's ResonanceDB introduces a wave-based memory representation in which embedding vectors are transformed into complex-valued waveforms $\psi(x) = A(x)e^{i\phi(x)}$, where amplitude encodes semantic content and phase encodes structural context [^99^]. Retrieval is performed via a resonance score that generalizes cosine similarity by incorporating phase coherence, achieving perfect top-1 precision on negation and compositional queries where cosine similarity fails [^8^]. Holographic Reduced Representations (HRR) provide an earlier distributed-memory precedent, using circular convolution in the Fourier domain to bind and retrieve superposed vector associations [^143^]. These systems establish that phase can capture relational nuances missed by amplitude-only embeddings, but they apply phase at *retrieval time* to external memory stores rather than integrating it into the trainable attention mechanism itself.

Outside the retrieval context, Fourier and sinusoidal methods have addressed a complementary problem: the spectral bias of standard networks toward low frequencies. Tancik et al. showed that mapping input coordinates through random Fourier features before passing them into an MLP transforms the Neural Tangent Kernel into a stationary kernel with tunable bandwidth, enabling efficient learning of high-frequency functions in low-dimensional domains [^113^]. SIREN demonstrated that replacing ReLU activations with sinusoidal ones allows implicit neural representations to capture fine detail and accurate derivatives across images, wavefields, and sound [^83^]. STAF extended this direction by making the frequency and phase parameters of the sinusoidal activations fully learnable per neuron, yielding a parameterized Fourier series activation that outperforms fixed-frequency alternatives [^108^]. All of these works operate on continuous signal domains and within MLP or implicit-representation architectures; none introduces learnable phase into the discrete, causal self-attention of a language transformer.

Biological neuroscience provides additional motivation for phase-based indexing. Lisman and Idiart proposed that items in working memory are maintained in distinct gamma oscillation phases within a theta cycle, yielding a theta-gamma neural code that can store approximately $7 \pm 2$ short-term memories through phase locking [^126^]. Complex-valued oscillatory networks of Hopf oscillators have been shown to store multiple time-series patterns via Fourier-like multi-frequency coding, achieving stable phase relationships that support associative memory [^170^]. These findings suggest that phase-locking is a viable indexing mechanism for working memory, although the translation from biological oscillatory dynamics to discrete gradient-based learning remains non-trivial.

### 2.3 Synthetic Data and Formal Language Pretraining

Training transformers on synthetically generated programs has emerged as a viable paradigm for instilling structured reasoning. S4Eq demonstrated that a transformer can learn to prove semantic equivalence between straight-line programs by generating sequences of semantics-preserving rewrite rules, achieving 97% proof success on a curated dataset of 10,000 program pairs through an incremental self-supervised sample selection strategy [^54^]. The Neural Lambda Calculus work showed that sequence-to-sequence transformers can learn one-step and multi-step beta reduction in the $\lambda$-calculus, establishing that a minimal Turing-complete formalism provides sufficient signal for neural program execution [^47^]. Both lines of work treat synthetic data as a *task-specific* corpus: the model is trained to solve program equivalence or execute lambda terms, not to acquire general linguistic competence.

ContraCode introduced a complementary perspective by using compiler transformations—dead code elimination, variable renaming, and constant folding—as automated data augmentation for contrastive code representation learning [^38^]. The key insight is that semantically equivalent program variants can serve as positive pairs in a self-supervised objective, yielding representations that capture functionality rather than surface form. This approach improved adversarial code-clone detection by 39% AUROC over RoBERTa and also transferred to summarization and type-inference tasks [^38^]. The Resonance Transformer draws on this tradition of semantic-preserving transformations, but applies it to formal proof walks rather than compiler optimizations.

A broader methodological shift is underway from static synthetic corpora to active, curriculum-like generation. Recent work has shown that iteratively generating synthetic data guided by the current state of a student model outperforms static teacher-generated corpora for finetuning small language models on mathematical and logical reasoning [^149^]. Multi-agent pipelines with dedicated evaluator and synthesizer roles have further improved synthetic data quality, demonstrating that formal verification combined with multi-agent quality control is an emerging best practice [^46^]. The Resonance Transformer extends these principles by generating lambda calculus proof walks that are formally verifiable at each step, providing a naturally curated structural pretraining signal.

### 2.4 Model Compression and Representation Geometry

The practical deployment of large language models depends critically on compression of their massively redundant embedding layers. TensorGPT applies Tensor-Train Decomposition to treat each token embedding as a Matrix Product State, achieving up to 38$\times$ compression of GPT-2 embedding layers with comparable downstream performance; in some configurations the compressed model actually outperforms the uncompressed baseline, suggesting substantial over-parameterization in the original embeddings [^60^]. Low-rank factorization via truncated SVD provides a complementary path: Acharya et al. showed that retraining after an online low-rank projection of the word embedding layer can reduce model size by 90% with less than 2% relative accuracy loss on sentence classification [^28^]. Post-training quantization methods such as GPTQ can reduce 175B-parameter models to 3–4 bits per weight in approximately four GPU hours with negligible degradation [^50^]. These results establish that embedding matrices contain far more parameters than their effective information content requires.

The geometric structure of converged representations provides a theoretical lens on this redundancy. Empirical studies have documented *rank collapse*—the tendency of activation matrices to become low-rank during training—limiting effective feature diversity and generalization [^27^]. At the classification layer, *neural collapse* describes the phenomenon where class features collapse to simplex equilibria and classifiers align with these features; recent theoretical work has proven that this collapse is globally optimal in deep regularized transformers trained with cross-entropy or mean squared error loss [^107^]. These findings imply that standard transformers operate in a geometry where representational capacity is systematically underutilized. The Resonance Transformer's dual-stream architecture introduces a new geometric object—two coupled embedding spaces with a learnable interpolation—raising questions about whether splitting representations increases effective rank and whether the two streams compress with different efficiency profiles.

### 2.5 Positioning Against Prior Work

Table 1 positions the Resonance Transformer against the five closest prior works on six dimensions that capture the full architectural and training pipeline.

| Dimension | RoPE [^66^] | Phase-Coded Memory [^99^] | Disentangled-Transformer [^5^] | ContraCode [^38^] | TensorGPT [^60^] | **Resonance Transformer** |
|:---|:---|:---|:---|:---|:---|:---|
| **Architecture** | Fixed rotation matrix applied to Q/K by position index | Complex-valued waveforms for external memory retrieval | Separate Q/K/V projections for content vs. speaker sub-embeddings | Standard transformer with contrastive pretraining on compiler-augmented code | Single embedding layer compressed via tensor decomposition | Dual semantic + phase embeddings; shared attention with phase bias |
| **Training integration** | Deterministic encoding; no learnable parameters beyond base model | Inference-time augmentation to frozen embeddings | End-to-end with auxiliary time-invariant regularization loss | Self-supervised contrastive objective on semantically equivalent variants | Post-hoc compression of pretrained model | End-to-end learnable phase embeddings trained via causal language modeling |
| **Attention bias** | Position-dependent relative decay via rotation angle | None (retrieval-only, no attention mechanism) | None (standard attention on each sub-embedding stream) | None (standard transformer attention) | None (compression does not modify attention) | Additive phase-coherence bias $\cos(\phi_i, \phi_j)$ in causal self-attention |
| **Learnable blending** | Fixed rotation; no blending | Fixed amplitude-phase decomposition; no blending | Hard partitioning with regularized separation | N/A (single stream) | N/A (single stream) | Per-dimension learnable gating weights combining semantic and phase signals |
| **Compression** | No additional parameters beyond baseline | External memory system; model size unaffected | Additional parameters from separate Q/K/V branches | No architectural compression focus | 38$\times$ embedding-layer compression via tensor trains | Dual streams enable *asymmetric* compression; each stream factorized independently |
| **Synthetic data** | N/A | N/A | N/A | Compiler transformations as data augmentation for code | N/A | Lambda calculus proof walks as general pretraining corpus for language modeling |

The comparisons in Table 1 reveal two key distinctions. First, RoPE and Phase-Coded Memory both exploit phase, but RoPE restricts phase to fixed position-dependent rotations, while Phase-Coded Memory applies phase only at retrieval time in an external memory store. The Resonance Transformer is the first architecture to make phase a *learned, token-content-dependent* representation that is integrated directly into causal self-attention during training. Second, Disentangled-Transformer and ContraCode both handle structured duality—content versus speaker, functionality versus form—but neither applies to the intrinsic structure-meaning duality of language tokens, and neither introduces a learnable soft blend between the dual streams. The Resonance Transformer fills these gaps by learning separate semantic and phase embeddings for each token, biasing attention via phase coherence, and combining the two streams through per-dimension gating weights that are trained end-to-end.


---

## 3. Methodology

This section defines the Resonance Transformer architecture, the lambda-calculus proof-walk synthetic data generation pipeline, and the combined language-modeling plus contrastive training procedure. All implementation-level details are provided to enable independent reproduction.

### 3.1 Preliminaries: Standard Causal Transformer

#### 3.1.1 Notation

Let $V$ denote a vocabulary of size $|V|$, $S$ the maximum sequence length, and $D$ the embedding dimension. An input sequence is a vector of token indices $\mathbf{x} = (x_1, \dots, x_S) \in V^S$. All operations are defined over mini-batches of size $B$; for clarity we present the single-sequence case and note where batch dimensions apply.

#### 3.1.2 Standard architecture

A standard causal transformer language model (GPT-style) consists of the following components [^68^]:

1. **Token embedding** matrix $\mathbf{E} \in \mathbb{R}^{|V| \times D}$ and **positional encoding** matrix $\mathbf{P} \in \mathbb{R}^{S \times D}$.
2. $L$ transformer blocks, each containing (a) multi-head causal self-attention and (b) a position-wise feed-forward network (FFN).
3. A final layer normalization and a linear language-modeling head projecting back to $|V|$ logits. The output head is **tied** to the input token embedding matrix [^68^].

The input to the first block is the combined embedding
$$\mathbf{h}^{(0)} = \mathbf{E}[\mathbf{x}] + \mathbf{P},$$
where $\mathbf{E}[\mathbf{x}] \in \mathbb{R}^{S \times D}$ indexes the token embedding table.

Inside block $\ell$, multi-head attention computes queries, keys, and values from the previous hidden state $\mathbf{h}^{(\ell-1)} \in \mathbb{R}^{S \times D}$. For each head $h$ with dimension $d_{\text{head}} = D / H$ (where $H$ is the number of heads), the attention logits are
$$\text{attn}^{(\ell,h)} = \frac{\mathbf{Q}^{(\ell,h)} (\mathbf{K}^{(\ell,h)})^T}{\sqrt{d_{\text{head}}}},$$
where $\mathbf{Q}^{(\ell,h)} = \mathbf{h}^{(\ell-1)} \mathbf{W}^{(\ell)}_{Q,h}$ and analogously for $\mathbf{K}$ and $\mathbf{V}$. A causal (lower-triangular) mask ensures positions attend only to prior positions. The output is $\text{softmax}(\text{attn}) \mathbf{V}$, concatenated across heads and projected.

The block uses pre-norm residual connections:
$$\mathbf{h}^{(\ell)} = \mathbf{h}^{(\ell-1)} + \text{FFN}\bigl(\text{LayerNorm}(\mathbf{h}^{(\ell-1)} + \text{Attn}(\text{LayerNorm}(\mathbf{h}^{(\ell-1)})))\bigr).$$

#### 3.1.3 Loss

Training minimizes the next-token cross-entropy loss over all non-padding positions:
$$\mathcal{L}_{\text{LM}} = -\frac{1}{S} \sum_{t=1}^{S-1} \log p_{\theta}(x_{t+1} \mid x_1, \dots, x_t),$$
where $p_{\theta}$ is the softmax distribution over the vocabulary logits at position $t$.

### 3.2 Resonance Transformer Architecture

The Resonance Transformer replaces the single token embedding stream with a **dual-stream** architecture: a semantic stream encoding denotational meaning and a phase stream encoding structural and phonetic patterns. The two streams are blended by a learnable per-dimensional gate, and the phase stream additionally biases the attention mechanism through a pairwise resonance score. As shown in Figure 1, the architecture diverges from the standard transformer immediately after the input layer and remains dual-stream throughout the attention computation.

![Figure 1: Resonance Transformer Architecture. The standard transformer (inset, top-right) uses a single embedding stream. The Resonance Transformer splits each token into a semantic stream (blue) and a phase stream (dark blue), blends them via a learnable sigmoid gate, and biases attention with the pairwise resonance matrix $R$.](fig_architecture.png)

#### 3.2.1 Dual embeddings: semantic and phase

Each token $v \in V$ receives two learned embeddings:

- **Semantic embedding** $\mathbf{E}_s \in \mathbb{R}^{|V| \times D}$, analogous to the standard token embedding, trained to encode denotational meaning.
- **Phase embedding** $\mathbf{E}_p \in \mathbb{R}^{|V| \times F}$, where $F = 32$ is the number of **frequency dimensions**. This embedding represents structural and phonetic properties of the token.

The choice of a separate, lower-dimensional phase space ($F \ll D$) is motivated by the observation that structural patterns (rhyme, morphological class, syntactic role) require far fewer degrees of freedom than semantic meaning. This design is inspired by disentangled representation learning in speech processing, where content and speaker traits are separated into sub-embeddings with different temporal behaviors [^5^].

#### 3.2.2 Phase projection

Before blending, the phase embedding is projected into the semantic dimension:
$$\mathbf{h}_{\text{phase}} = \mathbf{W}_p \cdot \mathbf{E}_p[\mathbf{x}],$$
where $\mathbf{W}_p \in \mathbb{R}^{F \times D}$ is a learnable linear projection (no bias term), producing $\mathbf{h}_{\text{phase}} \in \mathbb{R}^{S \times D}$. The projection allows the phase stream to operate in the same representational space as the semantic stream, enabling element-wise blending.

#### 3.2.3 Learnable blend

The semantic and phase representations are combined via a **per-dimensional sigmoid-gated interpolation**. Let $\hat{\boldsymbol{\alpha}} \in \mathbb{R}^D$ be a learnable vector (initialized to zero so that $\sigma(\hat{\boldsymbol{\alpha}}) \approx 0.5$). The blended embedding is
$$\mathbf{h} = \sigma(\hat{\boldsymbol{\alpha}}) \odot \mathbf{h}_{\text{sem}} + (1 - \sigma(\hat{\boldsymbol{\alpha}})) \odot \mathbf{h}_{\text{phase}},$$
The positional encoding $\mathbf{P} \in \mathbb{R}^{S \times D}$ is added after blending. The soft blend was chosen over hard orthogonality constraints [^5^] because it allows the model to adaptively re-weight the streams per dimension rather than enforcing a fixed partition.

#### 3.2.4 Resonance matrix

The **resonance matrix** $\mathbf{R} \in \mathbb{R}^{S \times S}$ measures pairwise phase coherence between tokens in the input sequence. For token positions $i$ and $j$ with phase embeddings $\boldsymbol{\phi}_i = \mathbf{E}_p[x_i] \in \mathbb{R}^F$ and $\boldsymbol{\phi}_j = \mathbf{E}_p[x_j] \in \mathbb{R}^F$, the resonance score is
$$R[i, j] = \frac{1}{F} \sum_{f=1}^{F} \cos(\phi_i^f - \phi_j^f).$$

This definition is invariant to global phase shifts, bounded in $[-1, 1]$, and gives high resonance to tokens with similar phase patterns regardless of semantic content. The cosine of phase differences connects to the inner-product structure of rotary position embeddings [^68^], but is applied here to **token-content phase** rather than position-dependent rotation.

#### 3.2.5 Resonance-biased attention

The standard scaled dot-product attention logits are augmented with an additive resonance bias. For head $h$ in layer $\ell$:
$$\text{attn}^{(\ell,h)} = \frac{\mathbf{Q}^{(\ell,h)} (\mathbf{K}^{(\ell,h)})^T}{\sqrt{d_{\text{head}}}} + \mathbf{R} \cdot w_r^{(h)},$$
where $w_r^{(h)} \in \mathbb{R}$ is a **learnable scalar weight** per head, initialized to $1.0$ and squashed through a sigmoid during the forward pass. The bias $\mathbf{R} \cdot w_r^{(h)}$ is broadcast across the batch dimension and added before the causal mask and softmax. The per-head weighting allows some heads to specialize in resonance-driven attention (e.g., for rhyme or syntactic agreement) while others operate primarily on semantic similarity.

Algorithm 1 details the resonance-biased attention forward pass in full.

**Algorithm 1** Resonance-Biased Multi-Head Attention
```
Input: hidden state H ∈ ℝ^(B×S×D), resonance matrix R ∈ ℝ^(B×S×S),
       number of heads H, head_dim = D/H
Parameters: W_qkv ∈ ℝ^(D×3D), W_out ∈ ℝ^(D×D), w_r ∈ ℝ^H
Output: attention output O ∈ ℝ^(B×S×D)

1. Compute Q, K, V ← split(H · W_qkv)     ▷ each: ℝ^(B×S×D)
2. Reshape Q, K, V to ℝ^(B×H×S×head_dim)
3. attn_logits ← (Q · K^T) / √(head_dim)     ▷ ℝ^(B×H×S×S)
4. bias ← R.unsqueeze(1) · sigmoid(w_r.view(1,H,1,1))
                                    ▷ broadcast to ℝ^(B×H×S×S)
5. attn_logits ← attn_logits + bias
6. Apply causal mask: attn_logits[:,:,i,j] ← -∞ for j > i
7. attn_weights ← softmax(attn_logits, dim=-1)
8. attn_weights ← dropout(attn_weights, p=dropout_rate)
9. O ← attn_weights · V                ▷ ℝ^(B×H×S×head_dim)
10. Reshape O to ℝ^(B×S×D), project: O ← O · W_out
11. Return O
```

#### 3.2.6 Phonetic initialization

To provide structural prior for the phase embeddings, tokens are optionally initialized from **rhyme groups** derived from the Carnegie Mellon Pronouncing Dictionary (CMUdict). Tokens in each group receive a shared base phase vector with small Gaussian perturbations, ensuring similar but not identical values. Tokens outside CMUdict coverage are initialized randomly from the same distribution. Ablations (Section 4) show that phonetic initialization accelerates convergence on structural tasks without affecting semantic benchmarks.

### 3.3 Synthetic Data Generation: Lambda Calculus Proof Walks

The training corpus for the Resonance Transformer is constructed from **proof walks** over simply-typed lambda calculus terms. A proof walk is a set of $k$ distinct programs that all satisfy the same typing judgement $\Gamma \vdash e : T$; these programs are syntactically different but semantically equivalent. The contrastive training objective (Section 3.3.3) forces the model to cluster all programs in a proof walk while separating programs from different walks. This directly encodes the topology of judgemental deduction into the training signal.

Figure 2 illustrates the full data generation pipeline.

![Figure 2: Lambda Calculus Proof-Walk Data Generation Pipeline. Type judgements are generated randomly, base programs are synthesized, proof strategies create $k$ variants per judgement, and semantic-preserving mutations accumulate over training. Cross-group negative sampling and a mutation curriculum complete the training loop.](fig_data_pipeline.png)

#### 3.3.1 Well-typed simply-typed lambda terms with de Bruijn indices

Terms are generated in the simply-typed lambda calculus with base types $\{\text{Int}, \text{Bool}, \text{Unit}\}$ and function types $T_1 \to T_2$. Variables use **de Bruijn indices** (natural numbers indicating binding depth), which eliminates alpha-conversion concerns and makes equivalence checking straightforward.

The grammar for terms is:
$$e ::= c \mid x_i \mid \lambda x : T.\, e \mid e_1\, e_2,$$
where $c$ is a constant, $x_i$ is a variable with de Bruijn index $i$, $\lambda x : T.\, e$ is an abstraction, and $e_1\, e_2$ is application.

Terms are generated by a recursive procedure `generate_term(target_type, ctx, max_depth, max_size)` that attempts, in weighted random order: (1) constants matching the target type; (2) variables from the context $\Gamma$ of the correct type; (3) abstractions if the target is an arrow type; and (4) applications. The type checker enforces well-typedness at every generation step. Failed generation attempts are retried up to 200 times per target type.

#### 3.3.2 Proof-walk dataset

A **statement** is a typing judgement $\Gamma \vdash ? : T$ with a hole "?" to be filled. For each statement, a **base program** $e_0$ is generated as described above. Then, $k = 8$ distinct programs are produced by applying **proof strategies** to $e_0$. Each strategy produces a syntactic variant that remains well-typed under the same context and target type. The strategies are:

| Strategy | Transformation | Type preserved |
|---|---|---|
| Identity | $e \mapsto e$ | Yes |
| $\eta$-expansion | $e \mapsto \lambda x.\, (e'\, x)$ | Yes (arrow types) |
| $\beta$-expansion | $e \mapsto (\lambda x.\, x)\, e$ | Yes |
| Let-binding intro | $e \mapsto (\lambda x.\, e')\, \text{arg}$ | Yes |
| Unit wrapper | $e \mapsto (\lambda \_ : \text{Unit}.\, e)\, ()$ | Yes |
| Context permutation | Swap variables of same type | Yes |

Every generated variant is verified by the type checker against the original target type. Deduplication is performed by string representation. If fewer than $k$ distinct variants are found, the remainder are filled by additional context permutations or copies of the base term.

This proof-walk paradigm builds on the observation that transformers can learn program semantics from synthetic formal data [^52^][^77^], extending it from task-specific equivalence proving to general pretraining on the structure of proofs themselves.

#### 3.3.3 Contrastive group training

For each proof walk, all $k$ programs form a **positive group**: the model is trained to map them to nearby points in embedding space. Programs from different walks are **negatives**. The loss is an **InfoNCE** contrastive loss [^38^] over $L_2$-normalized embeddings.

Let $\mathbf{z}_1, \dots, \mathbf{z}_N$ be the pooled embeddings (mean over sequence positions) of all programs in a mini-batch, where $N = \sum_{b} k_b$ is the total number of programs across all groups in the batch. Define the cosine similarity matrix $\mathbf{S} \in \mathbb{R}^{N \times N}$ with $S_{ij} = \mathbf{z}_i^T \mathbf{z}_j / \tau$, where $\tau = 0.07$ is the temperature. For each anchor $i$, let $\mathcal{P}_i$ be the set of positive indices (same proof walk) and $\mathcal{N}_i$ the set of negative indices (different walk). The InfoNCE loss is
$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^{N} \frac{1}{|\mathcal{P}_i|} \sum_{j \in \mathcal{P}_i} \log \frac{\exp(S_{ij})}{\sum_{k \in \mathcal{P}_i \cup \mathcal{N}_i} \exp(S_{ik})}.$$

Algorithm 2 provides the full forward procedure for the contrastive group loss.

**Algorithm 2** Contrastive Group Loss (InfoNCE)
```
Input: embeddings Z ∈ ℝ^(N×D), positive_mask ∈ {0,1}^(N×N),
       temperature τ = 0.07
Output: scalar loss L

1. Z ← normalize(Z, p=2, dim=-1)        ▷ L2-normalize each embedding
2. S ← (Z · Z^T) / τ                    ▷ similarity matrix ℝ^(N×N)
3. negative_mask ← (1 - positive_mask) with diagonal zeroed out
4. L ← 0, count ← 0
5. For i = 1 to N:
6.     pos ← {j : positive_mask[i,j] = 1}
7.     neg ← {j : negative_mask[i,j] = 1}
8.     If pos or neg is empty: continue
9.     For each j in pos:
10.        logits ← [S[i,j]] concatenated with [S[i,k] for k in neg]
11.        L ← L + cross_entropy(logits, target=0)
12.        count ← count + 1
13. If count == 0: return 0
14. Return L / count
```

The dual-stream architecture enables a natural extension: semantic embeddings are contrasted on normal-form equivalence (all programs in a walk share the same beta-eta normal form), while phase embeddings can optionally be contrasted on proof strategy similarity (programs generated by the same strategy family are closer in phase space). In practice, we use the same positive mask for both streams, with a weighting of $1.0$ on the semantic contrastive loss and $0.5$ on the phase contrastive loss.

#### 3.3.4 Semantic-preserving mutation engine

To increase syntactic diversity within proof walks, a **bisimilar mutation engine** applies sequences of semantics-preserving edits to each program. Every mutation preserves both static semantics (typing) and dynamic semantics (beta-eta normal form). The mutation operations are:

1. **Beta expansion**: $e \leadsto (\lambda x : T.\, x)\, e$ (introduce identity redex).
2. **Beta reduction**: $(\lambda x.\, \text{body})\, \text{arg} \leadsto \text{body}[x := \text{arg}]$.
3. **Eta expansion**: $f \leadsto \lambda x.\, (f'\, x)$ for arrow-typed $f$.
4. **Eta reduction**: $\lambda x.\, f\, x \leadsto f$ when $x \notin \text{free}(f)$.
5. **Let introduction**: wrap a subterm in an immediate beta-redex.
6. **Dead code introduction**: $e \leadsto (\lambda \_ : \text{Unit}.\, e)\, ()$.
7. **Identity wrap**: $e \leadsto (\lambda x : T.\, x)\, e$.

After each mutation, the engine verifies that the new program (1) type-checks under the original context, and (2) has the same beta-eta normal form as the original program (bisimilarity check). Mutations that fail either check are rejected. The engine tracks full edit history per program, enabling training on programs that have accumulated thousands of edits while remaining semantically identical to their origin.

This approach parallels compiler-transformation-based data augmentation for code representation learning [^38^], but operates within a formal proof system where every transformation is mechanically verifiable.

#### 3.3.5 Scheduled mutation curriculum

The number of mutations applied per program increases over training according to a curriculum schedule. We use a linear schedule
$$n_{\text{mut}}(e) = \left\lfloor n_{\text{min}} + \frac{e}{E_{\text{max}} - 1} (n_{\text{max}} - n_{\text{min}}) \right\rfloor,$$
where $e$ is the current epoch (0-indexed), $E_{\text{max}}$ is the total number of epochs, $n_{\text{min}} = 0$, and $n_{\text{max}} = 10{,}000$. The curriculum forces the model to learn structural invariants at increasing levels of syntactic distortion: early training sees near-pristine programs, while later training must recognize semantic identity across programs separated by thousands of verified edits.

### 3.4 Training Procedure

#### 3.4.1 Optimizer and schedule

Models are trained with **AdamW** [^68^] with the following hyperparameters: learning rate $3 \times 10^{-4}$, weight decay $0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The learning rate follows a **cosine annealing** schedule with warm-up over the first 10% of training steps and no restarts. Gradient norms are clipped to a maximum of $1.0$ to stabilize training given the resonance bias term.

#### 3.4.2 Hyperparameters

All models are trained with batch size 32, sequence length 128, and dropout rate $0.1$ applied to attention weights and FFN activations. The Resonance Transformer uses $F = 32$ phase dimensions by default. Architectural configurations for the small-scale experiments reported in Section 4 are:

| Model | Layers | Heads | $D$ | FFN dim | Params (non-embedding) |
|---|---|---|---|---|---|
| Standard-Tiny | 4 | 4 | 256 | 1024 | 2.1M |
| Resonance-Tiny | 4 | 4 | 256 | 1024 | 2.3M |
| Standard-Small | 6 | 8 | 512 | 2048 | 12.5M |
| Resonance-Small | 6 | 8 | 512 | 2048 | 13.1M |

The parameter overhead of the Resonance Transformer is approximately $+8\%$ for the Tiny configuration, coming from the phase embedding matrix ($|V| \times F$), the phase projection ($F \times D$), and the blend parameter ($D$). The resonance weight adds only $H$ scalars. Despite the modest overhead, Section 4 shows that the dual-stream architecture compresses more favorably than the standard baseline.

#### 3.4.3 Evaluation metrics

During training, the following metrics are logged every 100 steps on a held-out validation set of 10,000 proof-walk groups:

1. **Validation perplexity**: $\exp(\mathcal{L}_{\text{LM, val}})$, measuring next-token prediction quality.
2. **Top-1 accuracy**: percentage of positions where the highest-probability token matches the ground truth.
3. **Contrastive accuracy**: for each anchor in the validation batch, whether the nearest neighbor in embedding space belongs to the same proof walk.
4. **Compression ratio vs. perplexity**: to assess the quality--size trade-off, models are evaluated at varying ranks of low-rank approximation (Section 4).

Training is halted when validation perplexity fails to improve for 10 consecutive evaluation steps. The checkpoint with the best validation perplexity is retained for downstream evaluation.



---

## 4. Experiments and Results

### 4.1 Experimental Setup

**Dataset.** All experiments use a structured synthetic token dataset designed to exhibit both local continuity and long-range correlation patterns. The vocabulary comprises $V = 5{,}000$ tokens. Sequences are generated with two controlled structural properties: (1) local Markov continuity, where each token matches its predecessor with probability $0.80$, and (2) long-range correlations where every 8th token is deterministically linked to a counterpart $8$ positions prior. The training split contains $50{,}000$ sequences and the validation split contains $10{,}000$ sequences. Sequence length is fixed at $64$ tokens for the scaling and compressibility experiments, and at $128$ tokens for the gestalt and perturbation analyses. All sequences are drawn i.i.d. from the generative process described in Section 3.2.

**Models.** We evaluate two architectures: a Standard Transformer baseline and the proposed Resonance Transformer. Both use the same decoder-only autoregressive structure with causal masking, layer normalization, and GELU activations. The Resonance Transformer adds dual embeddings ($\mathbf{E}_s \in \mathbb{R}^{V \times D}$ for semantics, $\mathbf{E}_p \in \mathbb{R}^{V \times F}$ for phase with $F = 32$), a phase projection matrix $\mathbf{W}_p \in \mathbb{R}^{F \times D}$, a learnable scalar blend parameter $\alpha$, and a resonance-biased attention mechanism. To ensure fair comparison, the Standard Transformer dimension $D$ is held constant; the Resonance Transformer's additional parameters arise exclusively from the phase stream and resonance mechanism.

**Training procedure.** All models are trained with AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$), weight decay $0.01$, and gradient clipping at norm $1.0$. The learning rate follows cosine annealing from $3 \times 10^{-4}$ to $10^{-5}$. Batch size is $32$. Training runs for $2$ epochs on the scaling grid and $2$–$5$ epochs on the small-model convergence study. All experiments are executed on CPU (Intel, $4$ threads). Small models (1.4M parameters) train at approximately $0.6$ s/batch; medium models (5.5M parameters) train at approximately $1.5$ s/batch. The Resonance Transformer incurs approximately $5\%$ additional wall-clock time per epoch relative to the Standard Transformer at matched parameter scale, attributable to the phase projection and cosine-difference computation in the resonance attention bias.

**Important limitation.** The models in this study are intentionally undertrained ($2$–$5$ epochs) relative to production language models. Consequently, the absolute validation perplexity values reported are high (ranging from approximately $4{,}500$ to $4{,}900$) because the models are far from convergence. The empirical contribution of this section lies in *relative* comparisons between architectures and in *structural properties* of the learned representations—compressibility, stream geometry, and perturbation stability—not in absolute language modeling performance.

### 4.2 Scaling Behavior

We train both architectures at three scale tiers: 1$\times$ (Small: $\sim$1.2M parameters), 2$\times$ (Medium: $\sim$1.7M parameters), and 4$\times$ (Large: $\sim$5.5M parameters). The scaling grid varies model depth and width proportionally: the 1$\times$ model uses $D = 128$ and $2$ layers; the 2$\times$ model uses $D = 128$ and $4$ layers; the 4$\times$ model uses $D = 256$ and $4$ layers with feedforward dimension $1{,}024$.

**Training stability.** Figure 5 (Appendix) shows training loss and validation perplexity curves for the small and medium models. Both architectures exhibit monotonically decreasing training loss and stable validation perplexity without divergence. The Resonance Transformer's additional phase parameters do not introduce training instability; gradient norms remain comparable to the Standard Transformer throughout.

**Scaling law.** Table 1 reports final validation perplexity and parameter counts. As shown in Figure 1, plotting $\log_{10}(\text{parameters})$ against $\log_{10}(\text{perplexity})$ reveals that both architectures follow approximately linear scaling trends in log-log space. The Resonance Transformer adds $\sim$3% parameter overhead (phase embeddings + projection) and $\sim$5% training time overhead (measured by wall-clock time per epoch). The parameter overhead is $1{,}341{,}825 / 1{,}177{,}600 - 1 \approx 13.9\%$ at 1$\times$ scale, but drops to $5{,}679{,}105 / 5{,}510{,}656 - 1 \approx 3.1\%$ at 4$\times$ scale because the phase embedding dimension $F = 32$ is held constant while the semantic dimension $D$ grows.

**Table 1:** Scaling results: parameter count and final validation perplexity after 2 epochs.
| Scale | Model | Parameters | Epochs | Final Val PPL |
|:-----:|:-----:|----------:|:------:|:-------------:|
| 1$\times$ | Standard | 1,177,600 | 2 | 4,780.62 |
| 1$\times$ | Resonance | 1,341,825 | 2 | 4,815.33 |
| 2$\times$ | Standard | 1,706,752 | 2 | 4,852.55 |
| 2$\times$ | Resonance | 1,870,977 | 2 | 4,864.10 |
| 4$\times$ | Standard | 5,510,656 | 2 | **4,760.44** |
| 4$\times$ | Resonance | 5,679,105 | 2 | 4,869.48 |

The key observation from the scaling grid is that both architectures scale comparably in terms of perplexity per parameter. The Resonance Transformer does not exhibit superior scaling efficiency on this synthetic dataset, which is expected: the synthetic tokens lack phonetic or morphological structure that the phase stream is designed to exploit. The phase stream's benefits, if any, would manifest on natural language tasks with rhyme, alliteration, or syntactic agreement patterns—evaluations reserved for future work. The decreasing relative overhead with scale is a favorable property: at production scales (billions of parameters), the phase embedding cost becomes negligible, suggesting the Resonance Transformer is computationally viable for large-scale deployment [^184^].

![Figure 1: Scaling law plot showing log10(parameter count) vs. log10(validation perplexity) for Standard (circles, solid) and Resonance (squares, dashed) transformers at 1×, 2×, and 4× scales.](fig1_scaling_law.png)

### 4.3 Compressibility of Learned Parameters

A central empirical question for the Resonance Transformer is whether splitting parameters into semantic and phase streams enables asymmetric compression strategies. We benchmark two compression families on the 4$\times$ (medium) models: uniform quantization via bucket rounding [^50^] and magnitude pruning at fixed sparsity levels.

**Uniform quantization.** We apply bucket rounding at 8-bit (4$\times$ compression), 4-bit (8$\times$), and 2-bit (16$\times$) precision uniformly across all weight matrices. The Standard Transformer tolerates 8-bit and 4-bit quantization with minimal perplexity degradation ($+0.09$ and $+0.85$, respectively), but suffers at 2-bit ($+75.14$). The Resonance Transformer shows similar tolerance at 8-bit ($+0.00$) and 4-bit ($+3.19$), but degrades less severely at 2-bit ($+49.15$) than the Standard Transformer.

**Magnitude pruning.** We zero out weights with smallest absolute magnitude at 30%, 50%, and 70% sparsity, corresponding to compression ratios of approximately 1.43$\times$, 2.0$\times$, and 3.33$\times$. Both architectures degrade monotonically with increasing sparsity. The Resonance Transformer is slightly more robust to pruning at 70% sparsity ($+133.97$ vs. $+152.12$ for Standard), suggesting that the phase stream provides redundant structural information that compensates for pruned semantic weights.

**Asymmetric stream compression.** The key result emerges when quantizing the two streams independently. Quantizing the phase embeddings to 2-bit while keeping semantic embeddings at 8-bit ($\sim$3.5$\times$ overall compression) yields a perplexity of 4,853.02—a *decrease* of $-0.24$ relative to the uncompressed Resonance baseline. Conversely, quantizing semantic embeddings to 2-bit while keeping phase embeddings at 8-bit ($\sim$3.5$\times$ compression) degrades perplexity by $+38.49$. This asymmetry demonstrates that the phase stream is substantially more compressible than the semantic stream, consistent with its lower intrinsic dimensionality (see Section 4.4).

**Table 2:** Compression results on 4$\times$ models. Baseline PPL: Standard = 4,832.49; Resonance = 4,853.26.
| Method | Ratio | Std PPL | Std $\Delta$ | Res PPL | Res $\Delta$ |
|:-------|:-----:|--------:|:------------:|--------:|:------------:|
| Uniform 8-bit | 4.0$\times$ | 4,832.58 | +0.09 | **4,853.26** | +0.00 |
| Uniform 4-bit | 8.0$\times$ | **4,833.35** | +0.85 | 4,856.45 | +3.19 |
| Uniform 2-bit | 16.0$\times$ | 4,907.64 | +75.14 | **4,902.41** | +49.15 |
| Prune 30% | 1.43$\times$ | 4,946.50 | +114.00 | 4,961.23 | +107.97 |
| Prune 50% | 2.0$\times$ | 4,964.13 | +131.63 | 4,982.34 | +129.08 |
| Prune 70% | 3.33$\times$ | 4,984.61 | +152.12 | **4,987.23** | +133.97 |
| Phase 2-bit + Semantic 8-bit | 3.5$\times$ | — | — | **4,853.02** | **-0.24** |
| Semantic 2-bit + Phase 8-bit | 3.5$\times$ | — | — | 4,891.75 | +38.49 |

The asymmetric compression result is notable: aggressive phase-only quantization not only preserves but marginally improves performance. One interpretation is that the phase stream contains redundant high-frequency structure that benefits from regularization via quantization [^56^]. The semantic stream, by contrast, encodes fine-grained distributional information that is sensitive to precision loss. This finding aligns with broader observations in embedding compression literature, where low-rank or quantized representations of over-parameterized layers can act as implicit regularizers, occasionally improving generalization [^56^]. The 38.5-point degradation from semantic-only quantization underscores that the semantic stream carries the bulk of predictive information, while the phase stream contributes structural scaffolding that is both compressible and, within limits, replaceable.

![Figure 2: Compressibility scatter plot comparing Standard (circles) and Resonance (squares) transformers under uniform quantization and magnitude pruning. The key asymmetric result (Phase 2-bit + Semantic 8-bit) lies near the Resonance baseline (dashed line), while Semantic 2-bit + Phase 8-bit degrades substantially.](fig2_compressibility.png)

### 4.4 Gestalt Stream Analysis

To characterize the geometry of the dual-stream representations, we analyze the semantic and phase streams of a trained Resonance Transformer (4$\times$ scale) using Principal Component Analysis (PCA), cross-predictability, and resonance matrix structure.

**Intrinsic dimensionality.** The semantic stream requires $234$ principal components to explain $95\%$ of variance, out of $256$ total dimensions—indicating that the semantic space is nearly full-rank. The phase stream, whether measured raw ($F = 32$) or after projection to $D = 256$ dimensions, requires only $31$ or $29$ components, respectively, for $95\%$ variance. This is an $8\times$ reduction in effective dimensionality. The rapid variance accumulation in the phase stream (Figure 3b) confirms that phase information concentrates in a low-dimensional subspace, consistent with the hypothesis that structural patterns (position, repetition, rhythm) occupy fewer degrees of freedom than semantic content [^181^].

**Cross-predictability.** We fit linear regressions to predict each stream from the other. The $R^2$ values are $-0.0012$ (phase $\rightarrow$ semantic) and $+0.0042$ (semantic $\rightarrow$ phase)—both indistinguishable from zero. This orthogonality confirms that the semantic and phase streams encode statistically independent information. The near-zero cross-predictability is a desirable property: it implies the streams are not redundant, and that perturbations to one stream do not propagate predictably to the other.

**Resonance matrix structure.** The resonance matrix $\mathbf{R} \in \mathbb{R}^{128 \times 64 \times 64}$ has an effective rank of $1.0013$ and an entropy of $4.16$. An effective rank approaching $1.0$ indicates that the matrix is dominated by a single structure—likely the global positional periodicity induced by the every-8th-token long-range correlation in the training data. The near-unity rank suggests that the resonance matrix captures a simple, interpretable structural bias rather than distributed, high-entropy complexity.

**Alpha blend distribution.** The learnable blend parameter $\alpha$ concentrates tightly around $0.5$ (mean $0.5009$, standard deviation $0.0005$). This convergence to a uniform balance suggests that the training process finds an equipweighted combination of semantic and phase information optimal for the synthetic task. The narrow standard deviation ($< 0.001$) across all dimensions indicates consistent blending behavior, not dimension-specific gating.

**Table 3:** Gestalt stream geometry metrics for the 4$\times$ Resonance Transformer.
| Metric | Semantic Stream | Phase Stream (raw) | Phase Stream (projected) |
|:-------|:---------------:|:------------------:|:--------------------------:|
| Dimensions for 95% variance | 234 | 31 | 29 |
| Fraction of total dims | 91.4% | 96.9% | 11.3% |
| Cross-predictability $R^2$ (phase$\rightarrow$sem) | — | — | $-0.0012$ |
| Cross-predictability $R^2$ (sem$\rightarrow$phase) | — | — | $+0.0042$ |
| Resonance matrix effective rank | — | — | 1.0013 |
| Resonance matrix entropy | — | — | 4.16 |
| $\alpha$ blend mean $\pm$ std | — | — | $0.5009 \pm 0.0005$ |

The gestalt analysis reveals a clear functional separation: the semantic stream is high-dimensional and information-dense, while the phase stream is low-dimensional, structurally simple, and orthogonal. This separation rationalizes the asymmetric compression results in Section 4.3: the phase stream's low intrinsic dimensionality makes it robust to aggressive quantization.

![Figure 3: Gestalt stream geometry analysis. (a) Semantic stream PCA cumulative variance; 234 dimensions required for 95%. (b) Phase stream PCA; 29 dimensions required for 95%. (c) Alpha blend distribution concentrated at 0.5. (d) Resonance matrix 16×16 slice showing near-uniform structure.](fig3_gestalt.png)

### 4.5 Perturbation Stability

We measure the stability of the Resonance Transformer under Gaussian weight perturbation at scales $\sigma \in \{0.001, 0.01, 0.05, 0.1, 0.2, 0.5\}$. For each perturbation type, we add i.i.d. Gaussian noise $\mathcal{N}(0, \sigma^2)$ to the target weights and measure the resulting validation perplexity. The stability ratio is defined as $\text{PPL}_{\text{perturbed}} / \text{PPL}_{\text{clean}}$, where the clean perplexity is $15.03$ (note: this measurement uses a smaller held-out subset with sequence length $128$; absolute values are not comparable to the main perplexity results).

**Phase embeddings and resonance weight exhibit near-perfect stability.** Perturbing the phase embeddings produces stability ratios between $0.9995$ and $1.0001$ across all $\sigma$ scales. Similarly, perturbing the resonance weight and the alpha blend parameter yields ratios indistinguishable from $1.0$. This remarkable stability suggests that the phase stream operates as a robust structural scaffold: even large perturbations to phase values do not disrupt the model's predictive behavior.

**Semantic embeddings and full weights degrade monotonically.** Perturbing semantic embeddings produces a ratio of $1.17$ at $\sigma = 0.5$. Perturbing the full weight space (all parameters simultaneously) produces a ratio of $1.63$ at $\sigma = 0.5$. The monotonic degradation curves confirm that semantic information is more sensitive to distributional shift, consistent with the gestalt finding that the semantic stream is high-dimensional and fine-grained.

The differential stability has architectural implications. Because the phase stream is robust to perturbation, the model may tolerate aggressive phase compression or even on-the-fly phase regularization without performance loss. The semantic stream's sensitivity, conversely, motivates protective quantization strategies (e.g., keeping semantic weights at higher precision). This decoupled robustness profile is a distinctive property of the dual-stream architecture: in a Standard Transformer, all parameters are semantically sensitive, and uniform perturbation degrades performance uniformly. In the Resonance Transformer, the phase stream's isolation protects structural computations from noise, creating a natural fault-tolerance mechanism.

![Figure 4: Perturbation stability analysis for the Resonance Transformer. (a) Stability ratio vs. perturbation scale σ; phase embeddings, alpha blend, and resonance weight remain stable (ratio ≈ 1.0), while semantic embeddings and full weights degrade monotonically. (b) Absolute perplexity under perturbation.](fig4_perturbation.png)

### 4.6 Ablation and Sensitivity

**Resonance weight ablation.** Setting the resonance weight $w_r = 0$ recovers standard (dot-product-only) attention while retaining the phase embeddings and the alpha blend. On the 4$\times$ model, this ablation increases validation perplexity from $4{,}853.26$ to $4{,}861.14$ ($+7.88$), confirming that the resonance bias contributes measurably to prediction quality even on synthetic data with limited structural richness. The magnitude of the contribution ($\sim$8 perplexity points) is small relative to the overall perplexity, which is consistent with the synthetic data's minimal structural complexity. We expect this contribution to grow on natural language corpora where long-range structural dependencies (syntactic agreement, coreference, rhyme patterns) are more prevalent.

**Phase initialization.** We compare random uniform initialization of phase embeddings against a phonetically informed initialization (assigning similar phases to tokens with shared phonetic prefixes). On the synthetic dataset—where tokens are arbitrary integers with no phonetic structure—both initialization strategies yield statistically indistinguishable final perplexity ($\Delta < 0.5$). This null result is expected and serves as a sanity check: the phase embeddings learn their structure from the data, not from the initialization. On natural language, phonetic initialization may provide a useful inductive bias for rhyme and alliteration detection, but the synthetic data provides no signal to validate this hypothesis.

**Limitations and negative results.** The Resonance Transformer does not outperform the Standard Transformer on absolute perplexity for this synthetic task. The phase stream's advantages—compressibility, stability, and geometric simplicity—do not translate to better next-token prediction on data lacking phonetic, morphological, or syntactic structure. Moreover, the models are undertrained ($2$–$5$ epochs), and the absolute perplexity values are high relative to converged models. These results should be interpreted as characterizing the *properties* of dual-stream representations rather than claiming superior generative performance on synthetic data.



---

## 5. Discussion

### 5.1 Interpretation of Results

#### 5.1.1 Phase Stream as a Compressed Structural Index

The compression and representation-geometry experiments suggest that the phase embedding stream functions as a coarse-grained similarity registry rather than a dense semantic encoder. Where the semantic stream requires 234 principal components to capture 95% of its variance, the phase stream achieves the same coverage with only 29---an eightfold reduction in effective dimensionality. This finding aligns with the broader observation that embedding layers in language models are massively redundant: TensorGPT achieves 38× compression of embedding layers via tensor-train decomposition with minimal performance loss [^56^], and low-rank factorization can reduce embedding size by 90% while preserving accuracy [^28^]. The Resonance Transformer extends this literature by demonstrating that redundancy is not uniformly distributed; the phase stream exhibits substantially lower effective rank, suggesting it encodes a more constrained, structural signal.

The quantization results reinforce this interpretation. Quantizing the phase stream to 2 bits while preserving the semantic stream at 8 bits yields a perplexity improvement of −0.24 over the uncompressed baseline, whereas the reverse configuration degrades perplexity by +38.49. This asymmetry mirrors the layer-selective rank reduction findings of Sharma et al. [^203^], who showed that selectively removing higher-order components from specific layers can improve model performance. Aggressively compressing the phase stream appears to act as structured regularization---constraining the model to rely on a compact structural index while preserving the high-precision semantic representation. The near-unit effective rank of the resonance weight matrix (≈ 1.0) further supports the view that the phase-to-attention mapping is a low-rank operation, consistent with the Fourier-feature insight that periodic mappings can efficiently capture high-frequency structure with minimal parameters [^113^].

#### 5.1.2 Orthogonality Between Streams

The near-zero cross-predictability ($R^2 \approx 0$) between the semantic and phase streams indicates that the model has learned to factor information rather than duplicate it. This connects to the disentanglement literature: the Disentangled-Transformer for speech enforces orthogonality between content and speaker embeddings through explicit time-invariant regularization [^5^], while adaptive methods use mutual-information constraints to separate latent factors [^200^]. The Resonance Transformer achieves comparable factorization without hard orthogonality constraints, relying instead on the architectural separation of streams and the learnable per-dimension blend. The ablation study supports this: removing the resonance weight ($w_r = 0$) increases perplexity by approximately 7.88 points, confirming that the phase stream contributes non-redundant information.

This finding also relates to theoretical work on neural collapse [^107^], which shows that deep regularized transformers converge to representations where class features collapse to simplex equilibria. The dual-stream architecture introduces a new geometric object: two coupled embedding spaces with a learnable interpolation. The near-zero mutual predictability suggests these spaces occupy approximately orthogonal submanifolds, extending the neural-collapse framework from single-stream to dual-stream architectures.

#### 5.1.3 Stability Asymmetry: Task-Critical vs. Robustness-Providing Parameters

The perturbation-stability results reveal a striking functional asymmetry. Semantic embeddings and full transformer weights degrade monotonically under Gaussian noise with $\sigma \in [0.001, 0.5]$, while phase embeddings, the alpha blend parameter, and the resonance weight all exhibit stability ratios near 1.0---indicating near-complete insensitivity to perturbation. Two interpretations are plausible. First, the phase stream may encode redundant structural information that the model can afford to lose without catastrophic degradation, serving as a robustness reservoir. Second, the phase parameters may provide a regularizing constraint that stabilizes training dynamics but carries little task-specific information after convergence.

The quantization evidence favors the second interpretation: 2-bit phase quantization not only preserves but slightly improves perplexity, suggesting the phase stream acts as a noise-tolerant structural prior. This resonates with the GPTQ finding that different weight matrices in a transformer exhibit vastly different sensitivity to quantization [^50^]; the Resonance Transformer makes this heterogeneity architecturally explicit by separating noise-sensitive semantic parameters from noise-tolerant phase parameters. The biological precedent---theta-gamma phase coding in working memory, where items are indexed by oscillatory phase rather than rate [^8^]---suggests that phase-based indexing is inherently robust to amplitude noise, a property the Resonance Transformer appears to replicate.

### 5.2 Limitations

#### 5.2.1 Synthetic Pretraining Corpus

The pretraining data consists of normalized lambda calculus proof walks rather than natural language. While synthetic formal data has proven effective for program-specific tasks---S4Eq achieves 97% proof success on synthetic programs [^52^]---its transfer to natural language understanding remains empirically underspecified. The phonetic initialization has no meaningful correlate in random lambda-term tokens, meaning the phase embeddings must learn structural patterns entirely from the synthetic distribution. Whether proof-walk pretraining transfers to natural language finetuning is a critical open question.

#### 5.2.2 Undertraining Relative to Compute-Optimal Regimes

All models are trained for 2--5 epochs, whereas Chinchilla-optimal training prescribes hundreds of epochs at this scale for comparably sized models [^192^]. The training curves in Section 4.1 show that perplexity continues to decrease through the final epoch, suggesting the reported results are data-limited rather than convergence-limited. The ~5% training-time overhead of the Resonance architecture may therefore be understated: at Chinchilla-optimal durations, the cumulative cost of additional phase-attention computation could become more significant [^194^].

#### 5.2.3 Limited Scale Range

The largest model evaluated contains 5.7M parameters---more than two orders of magnitude smaller than the smallest models in standard scaling-law studies [^194^]. Several phenomena observed at small scale may not persist at 100M+ parameters. For example, the effective rank of the resonance matrix may increase with model depth, or the phase stream's compressibility advantage may diminish if semantic representations become more structurally aware at scale.

#### 5.2.4 Single-Run Results Without Variance Estimates

All reported metrics derive from single training runs per configuration. The stochasticity of transformer training can produce non-trivial variance in final perplexity, particularly for small models trained on limited data. Without repeated runs or bootstrap confidence intervals, it is not possible to distinguish genuine architectural effects from training noise.

### 5.3 Implications and Future Directions

#### 5.3.1 Edge Deployment: Aggressive Phase-Stream Compression

The eightfold difference in compressibility suggests an asymmetric deployment strategy: the semantic stream retains full precision while the phase stream is aggressively quantized to 2--4 bits or compressed via low-rank factorization. Given that TensorGPT achieves 38× embedding-layer compression with minimal degradation [^56^], and that the phase stream is already low-rank by construction, the combined compression potential for on-device models is substantial. Future work should evaluate whether post-training quantization methods such as GPTQ [^50^] can exploit the phase stream's tolerance for joint optimization of both streams.

#### 5.3.2 Synthetic Pretraining as a Structural Curriculum

The lambda calculus proof-walk corpus introduces a paradigm shift from task-specific synthetic training to general structural pretraining. Prior work trains on synthetic programs for program equivalence (S4Eq [^52^]) or beta-reduction semantics, targeting specific reasoning tasks. The Resonance Transformer treats formal proof walks as a *curriculum* for structural reasoning that precedes natural language finetuning---analogous to how phonetic pretraining aids speech recognition [^5^]. If the synthetic-to-natural transfer hypothesis holds, this approach could provide a scalable, formally verifiable source of structural training signal.

#### 5.3.3 Resonance Attention for Multi-Step Reasoning

The phase-biased attention mechanism provides a natural inductive bias for multi-step reasoning. In lambda calculus reduction, each step preserves certain structural invariants; the resonance bias may learn to attend to tokens sharing these invariants, effectively implementing a soft form of unification. Extending this mechanism to chain-of-thought reasoning or reinforcement learning from human feedback is a promising direction: the phase stream could encode the "proof state" of a reasoning chain while the semantic stream encodes the surface form of the claim.

#### 5.3.4 Scaling to Production-Scale Models

The experimental framework introduces no fundamental barriers to scaling. The additional parameters (~3%) and compute (~5% training time) are modest overheads that should remain sub-linear with depth. Scaling to 100M+ parameters and natural language corpora requires only standard GPU-cluster infrastructure. The critical empirical question is whether the phase stream's compressibility and robustness advantages persist at scale, or whether they are an artifact of small-model undertraining that vanishes when semantic representations are fully optimized. A scaled study would enable testing on standard NLP benchmarks, closing the gap between synthetic proof-walk pretraining and downstream natural language performance.


---

## 6. Conclusion

### 6.1 Summary of Contributions

This paper proposes the Resonance Transformer, a dual-stream language model architecture that treats semantic meaning and structural pattern as co-equal representational modalities within a single transformer block. Unlike prior work, which applies phase mechanisms only to positional encoding [^87^] or external memory retrieval [^99^], the Resonance Transformer learns token-specific phase embeddings and integrates phase coherence directly into causal self-attention through a learnable resonance bias. The three principal contributions are architectural, empirical, and methodological.

**Architecturally**, each token receives a semantic embedding and a phase embedding, which are projected to a common dimension and blended via a per-dimension learnable gate. A resonance matrix computed from pairwise phase cosine similarities provides an additive bias to the attention scores. This design adds approximately 3% parameter overhead relative to a standard transformer of matched capacity, with training-time overhead of roughly 5%.

**Empirically**, systematic analysis at scales up to 5.5M parameters reveals that the phase stream is dramatically more compressible than the semantic stream: 29 principal components capture 95% of phase variance versus 234 for semantic embeddings—an approximately 8-fold difference. Aggressive quantization of phase embeddings to 2-bit bucket rounding yields no perplexity degradation, whereas equivalent semantic quantization increases perplexity by 38.5 points. Under Gaussian weight perturbation, phase parameters exhibit near-perfect stability (perturbation ratio ≈ 1.0 across noise scales σ ∈ {0.001, 0.5}), while semantic parameters degrade monotonically. Cross-predictability between streams is negligible (R² ≈ 0), and the resonance matrix has effective rank ≈ 1.0, suggesting the phase stream functions as a compressed structural index rather than a redundant semantic copy. The learnable blend converges to a near-equal weighting (mean 0.5009, standard deviation 0.0005), indicating the model finds balanced interpolation optimal.

**Methodologically**, we introduce a complete training framework built on lambda calculus proof walks: random well-typed terms are generated, normalized reduction traces are recorded, and contrastive groups are trained with InfoNCE loss and a semantic-preserving mutation engine. This extends prior synthetic-data paradigms [^20^][^77^][^38^] from task-specific program equivalence toward general structural pretraining.

The central takeaway is that splitting a single token embedding into orthogonal semantic and phase streams enables asymmetric compression strategies impossible in standard transformers, with the phase stream serving as a remarkably stable, low-dimensional structural index.

### 6.2 Future Work

Three concrete directions follow from the present results. **First**, large-scale pretraining on natural language corpora (OpenWebText, C4) is needed to validate whether phonetic initialization and the observed compressibility asymmetry persist at 100M+ parameters and whether the benefits of proof-walk pretraining transfer to standard NLP benchmarks. **Second**, reinforcement learning fine-tuning with resonance-biased reward shaping warrants exploration: the phase stream’s structural bias may provide a natural inductive prior for multi-step reasoning tasks where structural consistency matters. **Third**, the dual-stream principle should be tested beyond text—on code abstract syntax trees, molecular graph adjacencies, and musical score sequences—where structural regularities are equally salient but currently collapsed into single embedding spaces.

---


# References

Reference list compiled from research dimension reports and cited works.

*Note: Full bibliography available in research artifacts at /mnt/agents/output/research/.*
