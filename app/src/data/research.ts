import {
  Binary,
  BrainCircuit,
  Braces,
  ChartNoAxesCombined,
  CircleHelp,
  Database,
  FileCode2,
  FlaskConical,
  GitCompareArrows,
  GitFork,
  GitPullRequestArrow,
  Landmark,
  Layers3,
  Microscope,
  Network,
  Route,
  Scale,
  SearchCheck,
  ShieldAlert,
  SlidersHorizontal,
  Split,
  Terminal,
  Waypoints,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

export type IconItem = {
  title: string;
  body: string;
  icon: LucideIcon;
};

export type EvidenceRow = {
  experiment: string;
  task: string;
  standard: string;
  bestResonant: string;
  interpretation: string;
  status: "strong signal" | "mixed" | "negative" | "early" | "running";
};

export type TaskRow = {
  name: string;
  source: string;
  signalTarget: string;
  whyItMatters: string;
  currentStatus: string;
};

export type VariantRow = {
  name: string;
  mechanism: string;
  whatItTests: string;
};

export const repoFacts = [
  { label: "Current thesis", value: "phase/structural stream may help topology-bearing sequence tasks" },
  { label: "Main caution", value: "current additive attention bias is not yet established as the key mechanism" },
  { label: "Best signal so far", value: "low-data BLiMP wh-island split, with phase variants ahead of standard" },
  { label: "Main blocker", value: "finding task regimes where structural information is necessary and learnable" },
  { label: "Repro style", value: "uv environment, scripted dataset fetches, pinned external cap-matcher source" },
  { label: "Compute targets", value: "local Apple MPS/CPU plus persvati AMD ROCm/CPU" },
];

export const thesisCards: IconItem[] = [
  {
    title: "The hypothesis",
    icon: Network,
    body:
      "Some sequence tasks contain durable structural invariants: binding, scope, graph topology, proof state, dependency paths, aliasing, or semantic role structure. A model with a dedicated phase-like stream may learn a compact coordinate system for those invariants instead of forcing every property through the same residual representation.",
  },
  {
    title: "The narrowed claim",
    icon: Scale,
    body:
      "The current evidence does not justify claims about general language modeling superiority. The live claim is narrower: phase-stream architectures are plausible structural priors, and the right evaluation is a battery of tasks where structure is required, surface cues are controlled, and ablations distinguish stream capacity from attention-bias effects.",
  },
  {
    title: "The research standard",
    icon: SearchCheck,
    body:
      "A result counts only if it survives iso-parameter baselines, phase-only and bias-only ablations, multiple seeds, hard negatives, and representation probes. Microvalidations are indicators; the target is a respectable literature task plus controlled synthetic tasks that reveal why a mechanism helps.",
  },
];

export const architectureSteps = [
  {
    label: "1. Token streams",
    detail:
      "Each token receives a normal semantic embedding and a lower-dimensional phase embedding. The phase stream can be static, dynamically updated, or injected into attention projections.",
  },
  {
    label: "2. Structural relation matrix",
    detail:
      "A kernel computes pairwise relations among phase vectors. The original version uses cosine phase differences; later variants include directional complex kernels, learned kernels, and structural-head paths.",
  },
  {
    label: "3. Interaction with attention",
    detail:
      "The initial implementation adds the relation matrix to attention logits. New experiments test whether this is the right interface, or whether Q/K conditioning, dedicated structural heads, or dynamic phase recurrence are better.",
  },
  {
    label: "4. Readout and ablation",
    detail:
      "Every experiment compares standard, iso-parameter standard, full resonance, phase-only, bias-only, inert-resonance, and newer variants. This is meant to separate representational benefit from extra parameters or accidental regularization.",
  },
];

export const architectureVariants: VariantRow[] = [
  {
    name: "standard",
    mechanism: "Ordinary transformer baseline.",
    whatItTests: "Whether the task is learnable without explicit structural machinery.",
  },
  {
    name: "standard_iso",
    mechanism: "Baseline widened to roughly match the extra resonance parameters.",
    whatItTests: "Whether apparent gains are just capacity or optimization budget.",
  },
  {
    name: "resonance_full_normalized",
    mechanism: "Semantic stream plus phase stream plus normalized relation bias.",
    whatItTests: "The original active architecture under a non-inert preset.",
  },
  {
    name: "phase_stream_only_normalized",
    mechanism: "Uses phase stream/projection without the additive relation bias.",
    whatItTests: "Whether the separate structural representation is enough by itself.",
  },
  {
    name: "bias_only_normalized",
    mechanism: "Injects relation bias without using phase projection as a representational stream.",
    whatItTests: "Whether pairwise attention bias is doing causal work.",
  },
  {
    name: "resonance_inert_normalized",
    mechanism: "Keeps architecture plumbing while suppressing active signal.",
    whatItTests: "Whether improvements come from implementation side effects.",
  },
  {
    name: "phase_dynamic_mlp / phase_dynamic_attn",
    mechanism: "Updates phase states across layers rather than treating token phase as a static lookup.",
    whatItTests: "Whether structural coordinates should evolve with context.",
  },
  {
    name: "phase_qk_film",
    mechanism: "Uses phase to modulate query/key projections.",
    whatItTests: "Whether phase is better used as attention geometry rather than as an additive logit prior.",
  },
  {
    name: "complex_directional",
    mechanism: "Adds asymmetric sine/phase terms instead of only symmetric cosine similarity.",
    whatItTests: "Whether directionality matters for proof steps, dependency arcs, and causal graph edges.",
  },
  {
    name: "structural_heads",
    mechanism: "Reserves attention heads for structural signals and lets semantic heads remain ordinary.",
    whatItTests: "Whether disentanglement should be architectural rather than shared through all heads.",
  },
];

export const evidenceRows: EvidenceRow[] = [
  {
    experiment: "BLiMP wh-island n=150",
    task: "Minimal-pair grammar judgment",
    standard: "71.1% acc",
    bestResonant: "79.3% acc, phase-only",
    interpretation:
      "Cleanest early signal. Full resonance and phase-only both beat standard, while bias-only and inert variants trail. This points more toward a useful phase stream than toward additive bias as the whole story.",
    status: "strong signal",
  },
  {
    experiment: "BLiMP wh-island n=100",
    task: "Lower-data grammar judgment",
    standard: "56.5% acc",
    bestResonant: "~66% acc",
    interpretation:
      "Same direction as n=150 but noisier. Needs more seeds and iso-param comparison before treating it as real.",
    status: "mixed",
  },
  {
    experiment: "BabyLM corpus LM",
    task: "Word-level next-token modeling",
    standard: "51.29 PPL",
    bestResonant: "51.35 PPL, full",
    interpretation:
      "No language-modeling win in this setup. Useful sanity check: the architecture should not be advertised as generally better until a real corpus run shows that.",
    status: "negative",
  },
  {
    experiment: "Cap matching smoke",
    task: "Knowledge-based unification classification",
    standard: "53.1% acc",
    bestResonant: "54.2% acc, phase-only",
    interpretation:
      "Generator is fast and self-contained. Smoke result only says the task runs; longer depth sweeps and hard negatives are underway.",
    status: "early",
  },
  {
    experiment: "Algebraic protocol smoke",
    task: "Protocol trace/term structure",
    standard: "60.4% acc",
    bestResonant: "61.5% acc, phase-only",
    interpretation:
      "Early hint that protocol-style algebraic structure is a useful task family, but too small to interpret.",
    status: "early",
  },
  {
    experiment: "Term unification fixed CPU",
    task: "First-order unification",
    standard: "75.6% acc, seed 201",
    bestResonant: "81.4% acc, standard_iso",
    interpretation:
      "Learnable and useful, but iso-param baseline is often strongest. This is exactly why iso-param controls are mandatory.",
    status: "mixed",
  },
  {
    experiment: "POJ-like problem classification",
    task: "Program classification",
    standard: "~25% val acc",
    bestResonant: "~37% phase-only in one run",
    interpretation:
      "Current formulation looks pathological or overfit-prone. Treat as diagnostic infrastructure, not evidence.",
    status: "running",
  },
];

export const syntheticTasks: TaskRow[] = [
  {
    name: "Cap matching",
    source: "Self-contained Python generator plus pinned reference source from emberian/reu_unif",
    signalTarget: "Knowledge-based unification under typed caps, substitutions, and incompatible constraints",
    whyItMatters:
      "It is close to Ember's earlier formal-methods work and naturally produces algebraic cases where surface form is not the semantic object.",
    currentStatus: "Implemented, smoke-tested, longer CPU/MPS sweeps running.",
  },
  {
    name: "Algebraic protocol traces",
    source: "Local synthetic generator",
    signalTarget: "Protocol state, role, nonce/key flow, and symbolic-message structure",
    whyItMatters:
      "Security-protocol notation provides structural dependencies that are not equivalent to local token prediction.",
    currentStatus: "Implemented and smoke-tested; needs richer adversarial negatives.",
  },
  {
    name: "First-order term unification",
    source: "Local generator",
    signalTarget: "Alpha-renaming, occurs-check-like conflicts, tree shape, and substitution consistency",
    whyItMatters:
      "Unification is a direct test of structural equivalence under variable binding rather than lexical overlap.",
    currentStatus: "Implemented; recursion bug fixed; now part of remainder sweeps.",
  },
  {
    name: "Graph walks with aliases",
    source: "Local graph generator",
    signalTarget: "Same latent graph rendered with different names and walks",
    whyItMatters:
      "Controls away entity-overlap shortcuts and forces graph-state inference.",
    currentStatus: "Implemented in probe suite; needs larger seed grid.",
  },
  {
    name: "Template equivalence with zero lexical overlap",
    source: "Local template generator",
    signalTarget: "Shared schema under disjoint vocabularies",
    whyItMatters:
      "A pure test of whether the model can learn structural type without lexical crutches.",
    currentStatus: "Implemented in first-wave sweeps.",
  },
  {
    name: "Causal graph intervention stories",
    source: "Planned extension of semantic-story generator",
    signalTarget: "Counterfactual answer depends on graph topology, not temporal order",
    whyItMatters:
      "Natural-language bridge task: same prose style, different causal graph.",
    currentStatus: "Planned next generator.",
  },
  {
    name: "Structural paraphrase / hard negatives",
    source: "Semantic graph-to-story pipeline",
    signalTarget: "Same graph vs one-edge-flipped hard negative with overlapping entities",
    whyItMatters:
      "Directly addresses the failure mode where same-graph retrieval saturates through word overlap.",
    currentStatus: "Prototype exists; needs stronger negative verifier.",
  },
];

export const datasetRows: TaskRow[] = [
  {
    name: "BLiMP",
    source: "alexwarstadt/blimp",
    signalTarget: "Minimal-pair syntactic and semantic acceptability",
    whyItMatters:
      "Small, fast, interpretable, and already showing the cleanest early signal on wh-island.",
    currentStatus: "Downloaded/usable; wh-island probes active.",
  },
  {
    name: "Linzen agreement",
    source: "TalLinzen/rnn_agreement",
    signalTarget: "Subject-verb number agreement with attractors",
    whyItMatters:
      "Canonical long-range syntax benchmark where structure should matter.",
    currentStatus: "Identified for download/integration.",
  },
  {
    name: "HANS",
    source: "tommccoy1/hans",
    signalTarget: "Heuristic failures in natural-language inference",
    whyItMatters:
      "Tests whether structural priors reduce lexical/positional shortcuts.",
    currentStatus: "In dataset lake plan.",
  },
  {
    name: "MSGS",
    source: "nyu-mll/msgs",
    signalTarget: "Controlled generalization from surface cues to syntactic features",
    whyItMatters:
      "Designed exactly for shortcut-vs-structure diagnostics.",
    currentStatus: "In dataset lake plan.",
  },
  {
    name: "BabyLM",
    source: "cambridge-climb/BabyLM",
    signalTarget: "Small natural-language pretraining corpus",
    whyItMatters:
      "Useful for transfer after structural pretraining; not by itself a structural test.",
    currentStatus: "Available through Hugging Face.",
  },
  {
    name: "COGS / SLOG",
    source: "COGS-style semantic parsing and bingzhilee/slog",
    signalTarget: "Compositional generalization and semantic-role structure",
    whyItMatters:
      "Bridges synthetic formal structure to natural-language meaning representations.",
    currentStatus: "Partly scouted; needs clean runner.",
  },
  {
    name: "AMR 3.0",
    source: "LDC2020T02",
    signalTarget: "Sentence-to-graph semantics",
    whyItMatters:
      "The mature real-world graph target. It is the right later-stage test, not the first probe.",
    currentStatus: "Blocked on LDC access.",
  },
];

export const interpretabilityPlan: IconItem[] = [
  {
    title: "Stream ablations",
    icon: Split,
    body:
      "Run matched evaluations with semantic-only, phase-only, bias-only, and full paths. The first causal question is whether phase encodes useful structure at all; the second is where it enters the computation.",
  },
  {
    title: "Activation patching",
    icon: GitCompareArrows,
    body:
      "Patch phase states, semantic residuals, and relation matrices between matched positive/negative examples. This should reveal whether errors move with the structural stream or the lexical stream.",
  },
  {
    title: "SAEs and probes",
    icon: Microscope,
    body:
      "Train sparse autoencoders and linear probes separately on semantic embeddings, raw phase, projected phase, attention outputs, and final residuals. The point is not to assume linear features, but to measure which geometry each stream actually uses.",
  },
  {
    title: "Geometry diagnostics",
    icon: ChartNoAxesCombined,
    body:
      "Track effective rank, cluster purity, compression curves, perturbation stability, CKA between streams, and whether structural labels form neighborhoods in phase space.",
  },
  {
    title: "Counterfactual tasks",
    icon: GitPullRequestArrow,
    body:
      "Use hard negatives that preserve words and change structure. Interpretability claims are weak unless the task itself forces the model to distinguish structural from lexical evidence.",
  },
  {
    title: "Intrinsic interpretability",
    icon: BrainCircuit,
    body:
      "The Sparse CLIP and Minkowski-geometry lines are relevant: interpretability may be better built into training and matched to modality geometry, not bolted on as a post-hoc sparse-linear assumption.",
  },
];

export const reproducibilitySteps = [
  {
    title: "Set up Python environment",
    command: "uv sync",
  },
  {
    title: "Fetch pinned external references",
    command: "bash scripts/setup_external_deps.sh",
  },
  {
    title: "Run a structural probe",
    command:
      "uv run python resonance/structural_task_probe.py --task blimp --conditions standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized --device auto --output_dir resonance/outputs/reviewer_probe",
  },
  {
    title: "Run cap matching",
    command:
      "uv run python resonance/structural_task_probe.py --task cap_matching --n_train 4096 --n_val 1024 --conditions standard,standard_iso,resonance_full_normalized,phase_stream_only_normalized,phase_qk_film --output_dir resonance/outputs/cap_matching_reviewer",
  },
  {
    title: "Rebuild the lab notebook",
    command: "uv run python scripts/build_lab_notebook.py --root resonance/outputs --out resonance/outputs/lab_notebook_hyper_live",
  },
  {
    title: "Watch and sync remote runs",
    command: "INTERVAL_SECONDS=600 MAX_ROUNDS=72 bash scripts/watch_hyper_results.sh",
  },
];

export const openQuestions: IconItem[] = [
  {
    title: "What is the actual useful mechanism?",
    icon: CircleHelp,
    body:
      "The phase stream may be useful while the additive attention bias is not. The next experiments test alternative interfaces: Q/K modulation, dynamic phase updates, directional kernels, and structural heads.",
  },
  {
    title: "Which tasks have the right difficulty?",
    icon: SlidersHorizontal,
    body:
      "A task must be hard enough that lexical shortcuts fail, easy enough for small models to learn, and instrumented enough that hidden states show meaningful separation.",
  },
  {
    title: "Does structural pretraining transfer?",
    icon: Route,
    body:
      "The important bridge is not synthetic-task leaderboard wins. It is whether proof walks, unification, graph aliases, and cap matching improve sample efficiency or robustness on natural-language structural tasks.",
  },
  {
    title: "What would falsify the program?",
    icon: ShieldAlert,
    body:
      "If iso-parameter baselines consistently match or beat phase variants across hard structural tasks, and probes show no stable structural neighborhoods in phase space, this version of the hypothesis should be rejected or redesigned.",
  },
];

export const pageSummaries = [
  {
    title: "Architecture",
    icon: Layers3,
    body:
      "The model family, why the original attention-bias path is under suspicion, and the concrete variants now implemented.",
    href: "/architecture",
  },
  {
    title: "Experiments",
    icon: FlaskConical,
    body:
      "Current evidence table, how to read positive/null/mixed results, and why the strongest current claim is narrow.",
    href: "/experiments",
  },
  {
    title: "Tasks",
    icon: Database,
    body:
      "Synthetic generators, literature datasets, and the regime-probe criteria for a meaningful structural task.",
    href: "/tasks",
  },
  {
    title: "Interpretability",
    icon: Microscope,
    body:
      "Stream ablations, activation patching, SAEs, representation geometry, and intrinsic interpretability hooks.",
    href: "/interpretability",
  },
  {
    title: "Reproducibility",
    icon: Terminal,
    body:
      "Commands, scripts, external dependency pinning, output locations, and the current overnight-run workflow.",
    href: "/reproducibility",
  },
];

export const codeArtifacts = [
  {
    path: "resonance/models.py",
    description: "Standard and resonance transformer model implementations.",
  },
  {
    path: "resonance/structural_task_probe.py",
    description: "Unified runner for structural tasks and architecture conditions.",
  },
  {
    path: "resonance/synthetic/cap_matching.py",
    description: "Self-contained cap-matching-inspired synthetic data generator.",
  },
  {
    path: "resonance/synthetic/algebraic_protocol.py",
    description: "Synthetic algebraic/security-protocol trace generator.",
  },
  {
    path: "resonance/synthetic/unification.py",
    description: "First-order term unification task generator.",
  },
  {
    path: "scripts/run_hyper_persvati_second_wave.sh",
    description: "Remote AMD ROCm second-wave experiment launcher.",
  },
  {
    path: "scripts/run_hyper_local_mps_algebraic.sh",
    description: "Local Apple MPS algebraic task launcher.",
  },
  {
    path: "research/cap-matching-integration.md",
    description: "Notes on how the undergrad cap-matching algorithm enters the research plan.",
  },
  {
    path: "HYPEREXPERIMENT.md",
    description: "Handoff document for the current variant/evaluation sweep.",
  },
];

export const relationAwarePriorArt = [
  "Relation-aware self-attention and structural attention encodings are the closest architectural neighborhood.",
  "Semantic-preserving program contrast is prior art; this project should not frame it as novel.",
  "The possible contribution is the specific phase/structural-stream factorization, task suite, and causal analysis of whether it helps.",
  "Quantum and microtubule metaphors are not evidence. They may inspire vocabulary, but the empirical work must stand on ordinary ML terms.",
];

export const formalDataIdeas: IconItem[] = [
  {
    title: "Proof walks",
    icon: Waypoints,
    body:
      "Generate statements, walk proof/search spaces, and train contrastively over multiple proofs of the same judgment. This directly targets the topology of deduction rather than accidental syntax.",
  },
  {
    title: "Program equivalence",
    icon: FileCode2,
    body:
      "Use semantics-preserving transformations, compiler-normalized forms, and hard negatives that share identifiers while changing behavior.",
  },
  {
    title: "Unification families",
    icon: Braces,
    body:
      "Generate typed terms, substitutions, caps, and incompatible constraints. This gives exact labels and a knob for structural depth.",
  },
  {
    title: "Alias-controlled graphs",
    icon: GitFork,
    body:
      "Render the same latent graph under disjoint names and different walk orders. The task can be tuned from easy to impossible.",
  },
  {
    title: "Natural-language bridges",
    icon: Binary,
    body:
      "Convert graph/proof/protocol structure into prose, then evaluate whether structural pretraining transfers to BLiMP, MSGS, COGS, SLOG, and AMR-like tasks.",
  },
  {
    title: "Literature tasks",
    icon: Landmark,
    body:
      "Use BLiMP, Linzen agreement, HANS, MSGS, BabyLM, COGS/SLOG, and eventually AMR to avoid only validating on custom toy worlds.",
  },
];
