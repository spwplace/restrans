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

