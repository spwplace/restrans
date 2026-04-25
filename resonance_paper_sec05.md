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
