# Dimension 2: Synthetic Data & Program Synthesis for Language Models

> Research on generating synthetic training data from formal systems (lambda calculus, proof assistants, program equivalence), and learning program semantics via neural networks.

---

## 1. S4Eq: Self-Supervised Learning to Prove Equivalence Between Programs via Rewrite Rules
**Citation:** Kommrusch, Steve, Martin Monperrus, and Louis-Noel Pouchet. "Self-Supervised Learning to Prove Equivalence Between Programs via Rewrite Rules." arXiv:2109.10476 (2021). IEEE TSE (extended).  
**URL:** https://arxiv.org/abs/2109.10476

**Summary:** Proposes S4Eq, a transformer-based system that proves semantic equivalence between two programs by generating a sequence of semantics-preserving rewrite rules. Uses self-supervised sample selection: initial supervised training on synthetic program pairs, then incremental self-supervised training on challenging proofs found by broader search. Achieves 97% proof success on synthetic data and finds equivalence in real GitHub code.

**Relevance:** Directly relevant to the Resonance Transformer's synthetic data generation from formal languages. S4Eq demonstrates that transformers can be trained effectively on entirely synthetic, formally-generated data (grammar-based program generation + rewrite rules). The self-supervised sample selection technique is a powerful paradigm for scaling synthetic training.

---

## 2. Towards a Neural Lambda Calculus: Neurosymbolic AI Applied to the Foundations of Functional Programming
**Citation:** Lamb, Luis, et al. "Towards a Neural Lambda Calculus: Neurosymbolic AI Applied to the Foundations of Functional Programming." arXiv:2304.09276 (2023).  
**URL:** https://arxiv.org/abs/2304.09276

**Summary:** Investigates whether neural networks (specifically Transformers) can learn to execute lambda calculus programs. Teaches the model One-Step Beta Reduction (OBR) and Multi-Step Beta Reduction (MBR). Shows that the simplicity and Turing-completeness of lambda calculus make it a powerful formalism for neural program execution.

**Relevance:** Establishes that lambda calculus is a viable target for neural network learning. The Resonance Transformer's synthetic data generation from lambda calculus walks builds directly on this foundation—synthetic reduction traces can serve as high-quality pretraining data.

---

## 3. Lambda Calculus meets Machine Learning (Dissertation)
**Citation:** (UFRGS dissertation, 2023).  
**URL:** https://lume.ufrgs.br/bitstream/handle/10183/264011/001175975.pdf

**Summary:** Comprehensive study of teaching Transformers to perform lambda calculus reductions. Generates datasets of beta reduction steps and trains seq2seq transformers to predict reduction outputs. Explores the representational challenges and successes of learning symbolic computation.

**Relevance:** Provides the methodological details for generating synthetic lambda calculus training data. The Resonance Transformer's proof-walk synthetic data can extend this by generating not just reductions but entire proof traces with semantic annotations.

---

## 4. LAMBDABEAM: Neural Program Search with Higher-Order Functions
**Citation:** (arXiv:2306.02049, 2023).  
**URL:** https://arxiv.org/abs/2306.02049

**Summary:** Proposes LAMBDABEAM, which extends program synthesis to the lambda calculus with higher-order functions. Builds on CROSSBEAM and demonstrates neural-guided search over lambda calculus terms for programming-by-example tasks.

**Relevance:** Shows that neural models can effectively search and synthesize in lambda calculus space. The Resonance Transformer's synthetic data pipeline could incorporate similar neural-guided generation to produce diverse, meaningful lambda terms for training.

---

## 5. Contrastive Code Representation Learning (ContraCode)
**Citation:** Jain, Paras, et al. "Contrastive Code Representation Learning." EMNLP 2021 / arXiv:2007.04973 (2020).  
**URL:** https://arxiv.org/abs/2007.04973

**Summary:** Proposes a self-supervised pretraining task for learning code functionality (not form) via contrastive learning. Uses automated source-to-source compiler transformations (dead code elimination, variable renaming, constant folding) to generate semantically equivalent variants. Trains models to identify equivalent programs among distractors.

**Relevance:** Demonstrates that compiler transformations can serve as powerful data augmentation for program representation learning. The Resonance Transformer's synthetic lambda calculus data can use analogous semantic-preserving transformations to generate diverse training examples from a core set of terms.

---

## 6. Proving and Disproving Equivalence of Functional Programs
**Citation:** (LARA, EPFL; PLDI 2023).  
**URL:** https://lara.epfl.ch/~milovanc/papers/pldi23.pdf

**Summary:** Proposes automated functional induction for verifying correctness of functional programming assignments. Uses automated equivalence checking and clustering to efficiently process large numbers of student submissions. Emphasizes that proofs (not just counterexamples) are essential for correctness guarantees.

**Relevance:** Highlights the importance of proof-based verification in program equivalence. The Resonance Transformer's proof-walk synthetic data aims to train models on sequences of proof steps, similar in spirit to this work's focus on verifiable transformations.

---

## 7. Towards Active Synthetic Data Generation for Finetuning Language Models
**Citation:** (arXiv:2512.00884, 2026).  
**URL:** https://arxiv.org/abs/2512.00884

**Summary:** Proposes an iterative synthetic data generation framework where a student model actively guides which data a teacher LLM should generate. The student prioritizes data points that would most improve its performance, creating a curriculum-like synthetic data loop.

**Relevance:** Provides a modern framework for thinking about synthetic data quality. The Resonance Transformer's lambda calculus synthetic data could benefit from similar active/student-guided generation to ensure coverage of challenging proof patterns.

---

## 8. Empowering Math Problem Generation and Reasoning via Synthetic Data
**Citation:** (ACL/EMNLP 2025).  
**URL:** https://aclanthology.org/2025.emnlp-main.1223.pdf

**Summary:** Proposes a continual learning framework for LLM math problem generation and reasoning using synthetic data. Combines supervised fine-tuning, data synthesis via model self-play and multi-agent collaboration, and direct preference optimization (DPO). Uses three agents (two evaluators, one synthesizer) to ensure data quality.

**Relevance:** Demonstrates that multi-agent synthetic data generation with quality checks can produce training data competitive with human annotations. Similar multi-agent/self-play approaches could verify and filter synthetic lambda calculus proof data.

---

## 9. Explaining Representation by Mutual Information
**Citation:** (arXiv:2103.15114, 2021).  
**URL:** https://arxiv.org/abs/2103.15114

**Summary:** Decomposes model representations into total information, decision-related information, and redundant information using mutual information between layers. Provides a principled framework for understanding what different parts of representations encode.

**Relevance:** Useful for analyzing what the Resonance Transformer's dual embeddings actually encode. Can help verify that the semantic stream captures meaning-related information while the phase stream captures structural/phonetic information.

---

## Key Findings / Takeaways for Dimension 2

1. **Synthetic data from formal languages is a proven training paradigm.** S4Eq and the Neural Lambda Calculus work both demonstrate that transformers can be trained on entirely synthetic, formally-generated data with strong generalization to real-world code.

2. **Lambda calculus is particularly suitable** as a synthetic data source because it is Turing-complete yet syntactically minimal, and its reduction semantics provide natural "ground truth" sequences for seq2seq training.

3. **Self-supervised sample selection** (S4Eq) and **contrastive equivalence learning** (ContraCode) are powerful techniques for scaling synthetic training without human labels. The Resonance Transformer can adopt both: generating proof walks as positive pairs and non-equivalent terms as negatives.

4. **Active synthetic data generation** and **multi-agent quality control** are emerging best practices for ensuring synthetic data utility. The Resonance Transformer's proof-walk pipeline should incorporate curriculum-like generation and automated equivalence verification.

5. **Gap identified:** No prior work systematically uses lambda calculus proof walks as pretraining data for general language models. Existing work trains on code equivalence or lambda reductions as end tasks; the Resonance Transformer proposes using such formal-system walks as a pretraining corpus to instill structural reasoning.

6. **Program equivalence as a training signal** is well-explored for code representations, but under-explored for natural language structural patterns (rhyme, morphology, syntax). The Resonance Transformer bridges this gap by treating language structure as a "program" to be learned.
