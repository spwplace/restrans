import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import {
  Github,
  Code2,
  Brain,
  Layers,
  GitFork,
  FlaskConical,
  Cpu,
  Zap,
  Target,
  GitBranch,
  Microscope,
  Puzzle,
  TestTube,
  Monitor,
} from "lucide-react";

const architectureStack = [
  {
    icon: GitFork,
    title: "9 Swappable Kernels",
    description:
      "Cosine, cosine-weighted, dot-product, RBF, Laplace, bilinear, complex magnitude, complex real, and attention-style kernels. Each computes pairwise similarity from phase embeddings and plugs into the same attention bias mechanism.",
  },
  {
    icon: Layers,
    title: "5 Phase Embedding Variants",
    description:
      "Real-valued learned tables, fixed Fourier features, complex-angle embeddings (with amplitude=1 and to_complex() integration), hierarchical multi-scale concatenation, and low-rank factorized embeddings.",
  },
  {
    icon: Zap,
    title: "6 Bias Application Modes",
    description:
      "Additive, multiplicative gate, residual gate, temperature-scaled, softmax-reweighted, and resonance-only (diagnostic). Each mode is a registered nn.Module that modifies how the resonance matrix enters attention logits.",
  },
  {
    icon: Target,
    title: "Per-Layer Configuration",
    description:
      "LayerConfigRegistry matches layer indices against patterns (all, even, odd, first:N, last:N, range:a:b) to assign custom kernels, bias modes, and initialization presets per layer. Lower layers can use local kernels (RBF) while upper layers use global ones (complex magnitude).",
  },
];

const modernArchs = [
  {
    name: "LLaMA",
    features: ["RMSNorm", "SwiGLU", "GQA", "RoPE"],
    variants: "Dense, MoE, Resonant Dense, Resonant MoE",
  },
  {
    name: "Qwen 3.5",
    features: ["Hybrid linear/full attention", "QK norm", "RoPE"],
    variants: "Dense, MoE, Resonant Dense, Resonant MoE",
  },
  {
    name: "Gemma 4",
    features: ["Dual attention", "PLE", "YOCO KV sharing"],
    variants: "Dense, MoE, Resonant Dense, Resonant MoE",
  },
];

const interpTools = [
  {
    icon: Microscope,
    title: "Sparse Autoencoders",
    description:
      "Trainable SAEs with ReLU encoder, MSE + L1 sparsity loss, dead-feature tracking, and reconstruction quality metrics. Used to analyze what the resonance stream encodes versus the semantic stream.",
  },
  {
    icon: Monitor,
    title: "Linear Probes",
    description:
      "Task-specific probes with cross-validation, early stopping, and k-fold evaluation. Compare interpretability across 7 extractable streams: semantic embed, phase raw, phase projected, residual streams, attention outputs, and final hidden states.",
  },
  {
    icon: Puzzle,
    title: "Intervention Hooks",
    description:
      "Context managers for phase perturbation (zero, permute, noise), resonance weight scaling, per-token phase row patching/copying, and full forward-pass tracing with resonance matrix capture at every layer.",
  },
];

const engineeringPractices = [
  {
    icon: TestTube,
    title: "Smoke Tests",
    description:
      "Comprehensive test suite covering all kernel × phase embedding combinations, all bias modes, per-layer registry matching, modern architecture forward passes, SAE/probe training, and hard-negative verification.",
  },
  {
    icon: Cpu,
    title: "Multi-Device Support",
    description:
      "Automatic device selection (CUDA, MPS, CPU). Tested on Apple M2 Max (MPS) and AMD Radeon 890M with ROCm 6.2 (gfx11 override). Handles MPS-specific operator fallbacks gracefully.",
  },
  {
    icon: Code2,
    title: "Clean Factory APIs",
    description:
      "Every swappable component uses a typed factory (build_kernel, build_phase_embedding, build_bias_mode) with consistent kwargs passing. No global config mutations; each component is self-contained and testable.",
  },
];

const reflections = [
  {
    title: "Architecture extraction is harder than it looks",
    body: "Porting inference-optimized models from vLLM's C++/CUDA codebase into clean PyTorch training code required stripping tensor parallelism, paged KV caches, and fused MoE kernels. Each model (LLaMA, Qwen, Gemma) took 500+ lines of careful reimplementation to preserve numerical behavior while making the code trainable and hackable.",
  },
  {
    title: "Vectorization dominates wall-clock time",
    body: "The first contrastive-loss implementation used Python for-loops over positive and negative pairs. Replacing it with fully vectorized InfoNCE sped training up by roughly two orders of magnitude. This reinforced a principle I now apply reflexively: profile the loss function before anything else.",
  },
  {
    title: "Default initialization can hide a mechanism",
    body: "With phase_init_std=0.3 and resonance_attn_weight=0.1, the attention delta from resonance is ~0.000027 — effectively invisible. This explained why early experiments showed null results. The architecture only becomes active when presets (wide, strong, normalized) increase these values by 1–2 orders of magnitude. This taught me to always verify that a proposed mechanism is actually exercising its code path.",
  },
  {
    title: "Synthetic data quality matters more than quantity",
    body: "Early evals on TinyStories word-level data and structured Markov sequences were too unstructured to test the architecture's core claim — that phase-structured attention helps with topology-bearing sequences. We halted experiments and pivoted to building better eval infrastructure (β-reduction traces, semantic graph hard negatives) before burning more compute.",
  },
];

export default function Home() {
  return (
    <div className="space-y-20 pb-12">
      {/* Hero */}
      <section className="relative overflow-hidden bg-gradient-to-b from-primary/5 to-background pt-16 pb-12">
        <div className="container mx-auto px-4 max-w-4xl text-center space-y-6">
          <Badge variant="secondary" className="text-xs">
            Self-Directed Study · Anthropic Fellows Program
          </Badge>
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight">
            Resonance Transformers
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            A from-scratch exploration of <strong>dual-stream attention mechanisms</strong>.
            Every token receives a semantic embedding and a phase embedding; pairwise
            phase similarities bias attention logits through a learnable, swappable,
            and per-layer-configurable resonance system.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3 pt-4">
            <a
              href="https://github.com/spwplace/restrans"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button size="lg" className="gap-2">
                <Github className="h-4 w-4" />
                View on GitHub
              </Button>
            </a>
          </div>
        </div>
      </section>

      {/* Project Overview */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold tracking-tight">What This Project Is</h2>
          <p className="text-muted-foreground mt-2 max-w-2xl mx-auto">
            Not a paper claiming results. A deep engineering study in building modular,
            interpretable, and extensible transformer variants from first principles.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <FlaskConical className="h-5 w-5 text-primary mb-2" />
              <CardTitle className="text-base">The Question</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground leading-relaxed">
              Standard transformers collapse all token properties — semantics,
              syntax, phonetics, structure — into a single embedding vector. What
              happens if we split them? Can a dedicated <em>phase stream</em> encode
              structural relationships (rhyme, bracket matching, scope) while a
              <em>semantic stream</em> handles denotation?
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <Cpu className="h-5 w-5 text-primary mb-2" />
              <CardTitle className="text-base">The Approach</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground leading-relaxed">
              Build the entire stack from scratch: base dual-stream transformer,
              9 swappable resonance kernels, 5 phase embedding variants, 6 bias modes,
              per-layer configuration registries, and faithful PyTorch reimplementations
              of LLaMA, Qwen3.5, and Gemma4 — each in dense, MoE, and resonant variants.
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <Brain className="h-5 w-5 text-primary mb-2" />
              <CardTitle className="text-base">The Scope</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground leading-relaxed">
              ~6,000 lines of PyTorch covering architecture, training harnesses,
              synthetic data generators (λ-calculus proof walks, semantic graph stories),
              interpretability tooling (SAEs, probes, intervention hooks), and a
              comprehensive smoke-test suite. All runnable from a single <code>uv</code> environment.
            </CardContent>
          </Card>
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* Modular Architecture Stack */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Layers className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">The Modular Stack</h2>
            <p className="text-muted-foreground">
              Every component is swappable, registered, and independently testable
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {architectureStack.map((item) => {
            const Icon = item.icon;
            return (
              <Card key={item.title} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-2">
                    <Icon className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base">{item.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {item.description}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Core Equations</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <p>
                The Resonance Transformer splits each token into two embeddings:
              </p>
              <ol className="list-decimal list-inside space-y-2">
                <li>
                  <strong>Dual embeddings</strong> —{" "}
                  <code className="bg-muted px-1 rounded">E_sem ∈ ℝ^(V×D)</code> +{" "}
                  <code className="bg-muted px-1 rounded">E_phase ∈ ℝ^(V×F)</code> (F = 32 by default).
                </li>
                <li>
                  <strong>Resonance matrix</strong> —{" "}
                  <code className="bg-muted px-1 rounded">R[i,j] = kernel(φ_i, φ_j)</code>{" "}
                  where the kernel is selected from the registry.
                </li>
                <li>
                  <strong>Attention bias</strong> — The chosen bias mode modifies
                  attention logits using <code className="bg-muted px-1 rounded">R</code>{" "}
                  and a per-head learnable weight.
                </li>
                <li>
                  <strong>Blend gate</strong> — A per-dimension sigmoid-blend
                  interpolates between semantic and phase projections before
                  entering the transformer blocks.
                </li>
              </ol>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Initialization Presets</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <p>
                Because the default preset (phase_std=0.3, weight=0.1) produces an
                attention delta of ~2.7×10⁻⁵ — effectively inert — the architecture
                ships with five initialization presets:
              </p>
              <ul className="list-disc list-inside space-y-1">
                <li>
                  <strong>default</strong> — (0.3, 0.1) for baseline comparisons
                </li>
                <li>
                  <strong>wide</strong> — (1.0, 0.3) for broader phase exploration
                </li>
                <li>
                  <strong>strong</strong> — (1.2, 1.0) for active resonance signal
                </li>
                <li>
                  <strong>very_strong</strong> — (2.0, 2.0) for diagnostic extremes
                </li>
                <li>
                  <strong>normalized</strong> — (0.3, 0.1) with per-head normalization
                </li>
              </ul>
              <p>
                Presets can be applied globally or per-layer via the registry,
                enabling systematic ablation of how much resonance signal is
                injected at different depths.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* Modern Architectures */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Cpu className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Modern Architecture Integration</h2>
            <p className="text-muted-foreground">
              Faithful PyTorch reimplementations extracted from production inference codebases
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {modernArchs.map((arch) => (
            <Card key={arch.name} className="hover:shadow-md transition-shadow">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">{arch.name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex flex-wrap gap-1.5">
                  {arch.features.map((f) => (
                    <Badge key={f} variant="secondary" className="text-xs">
                      {f}
                    </Badge>
                  ))}
                </div>
                <p className="text-sm text-muted-foreground">
                  <strong>Variants:</strong> {arch.variants}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-6 text-sm text-muted-foreground leading-relaxed">
          <p>
            Each architecture family is implemented in ~500 lines of plain PyTorch,
            stripped of inference-only optimizations (fused kernels, tensor parallelism,
            paged attention) and restructured for gradient-based training. The resonance
            mechanism is injected as a modular attention bias that works with GQA, RoPE,
            RMSNorm, and SwiGLU without altering the core backbone equations. All 14
            variants (standard + resonant × dense + MoE across 3 families) share a
            unified config dataclass and factory pattern.
          </p>
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* Interpretability & Analysis */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Microscope className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Interpretability & Analysis</h2>
            <p className="text-muted-foreground">
              Tools to understand what the resonance stream learns versus the semantic stream
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {interpTools.map((item) => {
            const Icon = item.icon;
            return (
              <Card key={item.title} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-2">
                    <Icon className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base">{item.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {item.description}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* Synthetic Data & Evaluation */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-primary/10 rounded-lg">
            <TestTube className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Synthetic Data & Evaluation</h2>
            <p className="text-muted-foreground">
              Topology-bearing datasets designed to test structural attention
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">β-Reduction Trace Dataset</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground leading-relaxed space-y-3">
              <p>
                A generator for lambda-calculus terms that produces verified
                reduction traces: each sample is a pair (original term, normal form)
                with the full β-reduction path recorded. Terms are checked for
                confluence and normal-form validity before inclusion.
              </p>
              <p>
                This dataset is designed to test whether a model can learn
                <em>rewrite topology</em> — the graph of which terms reduce to which
                — rather than surface statistical correlations.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Semantic Graph Hard Negatives</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground leading-relaxed space-y-3">
              <p>
                A story-generation pipeline that builds semantic graphs (entities,
                events, relations) and renders them into natural-language narratives.
                Each graph produces verified positive examples and
                <strong>hard negatives</strong> via 11 single-mutation operators:
              </p>
              <ul className="list-disc list-inside space-y-1">
                <li>reverse_edge, swap_target, flip_relation</li>
                <li>negate_event, swap_subject_object, change_quantifier</li>
                <li>remove_negation, change_location, weaken_cause</li>
                <li>swap_belief_holder, negate_belief_content</li>
              </ul>
              <p>
                All negatives are verified: no text collisions with positives,
                exactly one structural mutation per negative, and empirical
                mutation-type distribution tracking.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* Engineering Practices */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Code2 className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Engineering Practices</h2>
            <p className="text-muted-foreground">
              How the codebase is organized, tested, and run
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {engineeringPractices.map((item) => {
            const Icon = item.icon;
            return (
              <Card key={item.title} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-2">
                    <Icon className="h-5 w-5 text-primary" />
                    <CardTitle className="text-base">{item.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {item.description}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-3">Repository Structure</h3>
          <pre className="bg-muted rounded-lg p-4 text-sm overflow-x-auto font-mono">
{`resonance/
├── resonance/
│   ├── kernels.py              # 9 swappable resonance kernels
│   ├── phase_embeddings.py     # 5 phase embedding variants
│   ├── bias_modes.py           # 6 bias application modes
│   ├── layer_config.py         # Per-layer kernel/bias/preset registry
│   ├── config.py               # Central config dataclasses
│   ├── models.py               # Base Standard & Resonance transformers
│   ├── modern/                 # LLaMA, Qwen3.5, Gemma4 (dense + MoE + resonant)
│   ├── sae.py                  # Sparse autoencoder + trainer
│   ├── probes.py               # Linear probes + cross-validation trainer
│   └── interpretability.py     # Traces, interventions, hooks
├── synthetic/
│   ├── beta_reduction_dataset.py   # λ-calculus reduction trace generator
│   └── semantic_story.py           # Graph-to-story + 11 hard-negative mutations
├── tests/
│   └── test_smoke.py           # 11-test integration suite
├── experiment_text.py          # TinyStories training harness (14 architectures)
├── found_suite.py              # Unified foundational experiment runner
└── sweep_variants.py           # Grid sweep over kernel/phase/bias/preset combos`}
          </pre>
        </div>

        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">Quick Start</h3>
          <pre className="bg-muted rounded-lg p-4 text-sm overflow-x-auto font-mono">
{`# Clone and enter the environment
uv sync

# Run the full smoke-test suite
PYTHONPATH=. uv run python tests/test_smoke.py

# Train a resonant LLaMA on TinyStories
uv run python experiment_text.py --architecture resonant_llama \
  --resonance_kernel complex_magnitude --phase_embedding complex_angle \
  --bias_mode residual_gate --init_preset strong --epochs 20

# Run a β-reduction retrieval experiment
uv run python found_suite.py --experiment beta_reduction \
  --embed_dim 128 --n_layers 4 --epochs 10

# AMD GPU with ROCm gfx11 override
HSA_OVERRIDE_GFX_VERSION=11.0.0 python experiment_text.py ...`}
          </pre>
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* Process Reflections */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Brain className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Process Reflections</h2>
            <p className="text-muted-foreground">
              What worked, what didn't, and what I learned along the way
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {reflections.map((r) => (
            <Card key={r.title} className="hover:shadow-md transition-shadow">
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <GitBranch className="h-5 w-5 text-primary" />
                  <CardTitle className="text-base">{r.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {r.body}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* CTA */}
      <section className="container mx-auto px-4 max-w-4xl text-center pb-8">
        <h2 className="text-2xl font-bold tracking-tight mb-4">
          Explore the Code
        </h2>
        <p className="text-muted-foreground mb-6">
          The full repository, with instructions for reproduction, is available on GitHub.
          Every architecture variant, training harness, and analysis tool is runnable
          from a single <code>uv</code> environment.
        </p>
        <a
          href="https://github.com/spwplace/restrans"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Button size="lg" className="gap-2">
            <Github className="h-4 w-4" />
            github.com/spwplace/restrans
          </Button>
        </a>
      </section>
    </div>
  );
}
