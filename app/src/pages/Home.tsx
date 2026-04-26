import { Link } from "react-router";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import {
  BookOpen,
  Image,
  Download,
  Code2,
  Brain,
  Bot,
  ArrowRight,
  Zap,
  Shield,
  Layers,
  GitFork,
} from "lucide-react";

const highlights = [
  {
    icon: Layers,
    title: "8× More Compressible",
    description:
      "Phase stream requires only 29 principal components for 95% variance vs. 234 for semantics—enabling aggressive asymmetric quantization.",
    stat: "29 vs 234 PCs",
  },
  {
    icon: Zap,
    title: "3% Parameter Overhead",
    description:
      "At 5.7M parameters, the dual-stream design adds only ~3.1% parameters and ~5% wall-clock time versus standard transformers.",
    stat: "~3% overhead",
  },
  {
    icon: Shield,
    title: "Statistical Independence",
    description:
      "Cross-predictability between semantic and phase streams is negligible (R² ≈ 0), confirming the model learns genuinely separate representations.",
    stat: "R² ≈ 0",
  },
  {
    icon: GitFork,
    title: "Perturbation Stability",
    description:
      "Phase parameters remain stable (stability ratio ≈ 1.0) under Gaussian noise up to σ = 0.5 while semantic parameters degrade monotonically.",
    stat: "σ = 0.5 stable",
  },
];

export default function Home() {
  return (
    <div className="space-y-20 pb-12">
      {/* Hero */}
      <section className="relative overflow-hidden bg-gradient-to-b from-primary/5 to-background pt-16 pb-12">
        <div className="container mx-auto px-4 max-w-4xl text-center space-y-6">
          <Badge variant="secondary" className="text-xs">
            Research Paper · Dual-Stream Transformer Architecture
          </Badge>
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight">
            Resonance Transformers
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            Dual-stream architectures with phase-structured attention. Every token
            receives a <strong>semantic embedding</strong> and a{" "}
            <strong>phase embedding</strong>—enabling structurally similar tokens
            to attend more readily to one another.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3 pt-4">
            <Link to="/paper">
              <Button size="lg" className="gap-2">
                <BookOpen className="h-4 w-4" />
                Read the Paper
              </Button>
            </Link>
            <Link to="/figures">
              <Button size="lg" variant="outline" className="gap-2">
                <Image className="h-4 w-4" />
                View Figures
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Key Results */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold tracking-tight">Key Results</h2>
          <p className="text-muted-foreground mt-2">
            Systematic evaluation at scales up to 5.7M parameters
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {highlights.map((h) => {
            const Icon = h.icon;
            return (
              <Card key={h.title} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <Icon className="h-5 w-5 text-primary" />
                    <span className="text-xs font-semibold text-primary bg-primary/10 px-2 py-0.5 rounded-full">
                      {h.stat}
                    </span>
                  </div>
                  <CardTitle className="text-base mt-3">{h.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {h.description}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* For Programmers */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Code2 className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">For Programmers</h2>
            <p className="text-muted-foreground">The artifact, code structure, and how to run it</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">What This Is</h3>
              <p className="text-muted-foreground leading-relaxed">
                The Resonance Transformer is a PyTorch implementation of a dual-stream
                causal language model. It lives alongside a standard transformer baseline
                in the same codebase, so you can train both and compare. The repo includes
                a synthetic lambda-calculus proof-walk generator, training harnesses for
                three modes (standard, resonance, synthetic), and analysis scripts for
                compressibility, perturbation stability, and representation geometry.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-2">Quick Start</h3>
              <pre className="bg-muted rounded-lg p-4 text-sm overflow-x-auto font-mono">
{`# Train a resonance model at small scale
uv run python resonance/train.py --mode resonance --scale small

# Train standard baseline for comparison
uv run python resonance/train.py --mode standard --scale small

# Synthetic proof-walk training with mutations
uv run python resonance/train.py --mode synthetic --scale medium --mutate

# Reproduce exactly
uv run python resonance/train.py --mode resonance --scale small --seed 42`}
              </pre>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-2">Key Files</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex gap-2">
                  <code className="bg-muted px-1.5 rounded text-xs shrink-0">resonance/resonance/models.py</code>
                  <span>StandardTransformer and ResonanceTransformer classes</span>
                </li>
                <li className="flex gap-2">
                  <code className="bg-muted px-1.5 rounded text-xs shrink-0">resonance/train.py</code>
                  <span>Unified training harness with scale presets</span>
                </li>
                <li className="flex gap-2">
                  <code className="bg-muted px-1.5 rounded text-xs shrink-0">resonance/resonance/data.py</code>
                  <span>Synthetic structured dataset generator</span>
                </li>
                <li className="flex gap-2">
                  <code className="bg-muted px-1.5 rounded text-xs shrink-0">resonance/experiments/</code>
                  <span>Compressibility, gestalt, and perturbation analysis</span>
                </li>
              </ul>
            </div>
          </div>

          <Card className="h-fit">
            <CardHeader>
              <CardTitle className="text-base">Architecture at a Glance</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <p>
                The ResonanceTransformer subclasses a standard GPT-style decoder. The
                key differences are all in the input layer and attention mechanism:
              </p>
              <ol className="list-decimal list-inside space-y-2">
                <li>
                  <strong>Dual embeddings</strong> — Each token gets{" "}
                  <code className="bg-muted px-1 rounded">E_sem ∈ ℝ^(V×D)</code> and{" "}
                  <code className="bg-muted px-1 rounded">E_phase ∈ ℝ^(V×F)</code>{" "}
                  (F=32).
                </li>
                <li>
                  <strong>Phase projection</strong> —{" "}
                  <code className="bg-muted px-1 rounded">W_p ∈ ℝ^(F×D)</code> projects
                  phase into semantic dimension space.
                </li>
                <li>
                  <strong>Learnable blend</strong> — A per-dimension sigmoid gate{" "}
                  <code className="bg-muted px-1 rounded">α</code> interpolates between
                  the two streams.
                </li>
                <li>
                  <strong>Resonance matrix</strong> — Pairwise cosine similarities of
                  phase embeddings:{" "}
                  <code className="bg-muted px-1 rounded">R[i,j] = (1/F) Σ cos(φ_i^f − φ_j^f)</code>.
                </li>
                <li>
                  <strong>Resonance-biased attention</strong> — Attention logits get an
                  additive bias{" "}
                  <code className="bg-muted px-1 rounded">R · w_r^(h)</code> per head.
                </li>
              </ol>
              <p>
                Everything else—causal masking, pre-norm residuals, GELU FFNs—is
                identical to the standard transformer baseline.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* For ML Scientists */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Brain className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">For ML Scientists</h2>
            <p className="text-muted-foreground">Architecture, methodology, and empirical findings</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Architecture Motivation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground leading-relaxed">
              <p>
                Standard causal transformers collapse all token properties—semantic
                denotation, orthographic form, phonetic structure, syntactic role—into
                a single embedding vector. This forces an implicit coordinate system
                that must simultaneously separate meaning from sound and structure.
              </p>
              <p>
                The Resonance Transformer introduces a principled split: a{" "}
                <strong>semantic stream</strong> for denotational content and a{" "}
                <strong>phase stream</strong> for structural and phonetic pattern.
                The phase stream uses a lower-dimensional space (F=32 vs D=128–256)
                motivated by the observation that structural regularities require far
                fewer degrees of freedom than semantic meaning.
              </p>
              <p>
                A <strong>learnable per-dimension blend</strong> (sigmoid-gated
                interpolation) replaces hard orthogonality constraints, allowing the
                model to adaptively re-weight each stream. The{" "}
                <strong>resonance matrix</strong>—pairwise phase cosine
                similarities—biases attention scores additively, enabling structurally
                similar tokens to attend more strongly regardless of semantic content.
              </p>
            </CardContent>
          </Card>

          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Training Signal</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Models are trained on synthetic lambda calculus proof walks—not as a
                task-specific corpus, but as a <strong>general pretraining signal</strong>{" "}
                designed to instill structural reasoning. Contrastive group objectives
                and semantics-preserving mutations provide additional regularization.
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Compressibility Asymmetry</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Aggressive 2-bit quantization of phase embeddings{" "}
                <strong>improves</strong> validation perplexity by 0.24 points, while
                equivalent semantic quantization degrades it by 38.49. This suggests
                the phase stream functions as a compact structural index that benefits
                from regularization.
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Perturbation Stability</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Phase parameters exhibit near-unit stability ratios under Gaussian
                noise (σ ∈ [0.001, 0.5]), while semantic embeddings degrade
                monotonically. This decoupled sensitivity profile suggests dual-stream
                architectures may offer inherent robustness advantages.
              </CardContent>
            </Card>
          </div>
        </div>

        <div className="mt-8">
          <Link to="/paper">
            <Button variant="outline" className="gap-2">
              Read Full Methodology & Results
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* For AI Devs / Agent Wranglers */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Bot className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">
              For AI Devs & Agent Wranglers
            </h2>
            <p className="text-muted-foreground">How this project went, what worked, and what didn't</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">The Process</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground leading-relaxed">
              <p>
                This project started as a reproduction of an architecture sketch from a
                GitHub repo. The first stage was extracting clean, runnable PyTorch
                modules from a notebook-style codebase—turning research code into
                something that could be trained systematically at multiple scales.
              </p>
              <p>
                Once the baseline and resonance models were training stably, we
                implemented four novel experiment tracks: perplexity stability under
                perturbation, compressibility via quantization and pruning, gestalt
                stream geometry analysis, and a synthetic lambda-calculus proof-walk
                generator with contrastive training.
              </p>
              <p>
                The paper was written incrementally, with each section drafted after
                its corresponding experiments completed. Figures and tables were
                generated automatically from experiment outputs to avoid manual
                copy-paste errors.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Struggles & Lessons</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground leading-relaxed">
              <p>
                <strong>Undertraining is the biggest caveat.</strong> All models ran
                for only 2–5 epochs on CPU. Absolute perplexities are high (~4,800)
                because nothing is converged. The results are meaningful for
                <em> relative</em> comparisons and <em>structural properties</em> of
                representations, but not for absolute language modeling performance.
              </p>
              <p>
                <strong>Synthetic data transfer is unproven.</strong> The lambda
                calculus proof walks are an interesting pretraining signal, but we
                never validated transfer to natural language. The phonetic
                initialization also has no meaningful correlate in lambda-term tokens,
                so the phase embeddings must learn structure entirely from the
                synthetic distribution.
              </p>
              <p>
                <strong>Single-run variance is real.</strong> Every number comes from
                one training run. At small scales, training noise can easily swamp
                architectural differences. We report the numbers honestly, but
                confidence intervals would require repeated runs.
              </p>
              <p>
                <strong>GitHub Pages + SPA routing.</strong> This very website was a
                lesson in HashRouter versus BrowserRouter. Deploying a React SPA to a
                project page subdirectory requires either HashRouter or careful
                basename handling. We went with HashRouter for simplicity.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">What Worked Well</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground leading-relaxed">
              <p>
                The <strong>unified training harness</strong> with scale presets
                (tiny/small/medium/large) made it trivial to run systematic grids.
                Every run auto-saves config, reproducibility metadata, and checkpoints.
              </p>
              <p>
                The <strong>asymmetric compressibility finding</strong> is robust and
                surprising: quantizing phase embeddings to 2-bit <em>improves</em>{" "}
                perplexity. This wasn't hypothesized upfront—it fell out of the
                compression experiments cleanly.
              </p>
              <p>
                Using <strong>automated figure generation</strong> from experiment
                outputs meant the paper could be regenerated end-to-end whenever
                results changed. No manual plot updates.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Tooling & Stack</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground leading-relaxed">
              <p>
                <strong>Research:</strong> PyTorch, NumPy, Matplotlib, scikit-learn for
                PCA and effective rank. MPS auto-detection for Apple Silicon training.
              </p>
              <p>
                <strong>Experiments:</strong> Four analysis scripts
                (compressibility, gestalt, perturbation, param compressibility) that
                load checkpoints and emit JSON + PNG outputs.
              </p>
              <p>
                <strong>Paper:</strong> Markdown sections compiled into a single
                document, then converted to DOCX with Pandoc-style tooling.
              </p>
              <p>
                <strong>Website:</strong> Vite + React + TypeScript + Tailwind +
                shadcn/ui, deployed via GitHub Actions to GitHub Pages. React-markdown
                renders the paper in-browser.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      <Separator className="container mx-auto px-4 max-w-6xl" />

      {/* Downloads */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold tracking-tight">Downloads</h2>
          <p className="text-muted-foreground mt-2">
            Code, data, models, and the full paper
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-w-4xl mx-auto">
          <Card className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Download className="h-5 w-5 text-primary" />
                <CardTitle className="text-base">Full Package</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Complete codebase, datasets, checkpoints, and experiment outputs.
                (~100 MB — available via GitHub Releases)
              </p>
              <a href="https://github.com/spwplace/restrans/releases" target="_blank" rel="noopener noreferrer">
                <Button variant="outline" size="sm" className="w-full gap-2">
                  <Download className="h-4 w-4" />
                  Get on GitHub Releases
                </Button>
              </a>
            </CardContent>
          </Card>

          <Card className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Download className="h-5 w-5 text-primary" />
                <CardTitle className="text-base">Code Package</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Source code, models, training scripts, and experiment harness only.
                (~50 MB)
              </p>
              <a href="./resonance-package.tar.gz" download>
                <Button variant="outline" size="sm" className="w-full gap-2">
                  <Download className="h-4 w-4" />
                  Download
                </Button>
              </a>
            </CardContent>
          </Card>

          <Card className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <BookOpen className="h-5 w-5 text-primary" />
                <CardTitle className="text-base">Paper (DOCX)</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4">
                Formatted academic paper with endnotes and figures.
              </p>
              <a href="./Resonance_Transformers_Paper.docx" download>
                <Button variant="outline" size="sm" className="w-full gap-2">
                  <Download className="h-4 w-4" />
                  Download
                </Button>
              </a>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
}
