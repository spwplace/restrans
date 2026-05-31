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

export type PriorArtRow = {
  family: string;
  representativeWork: string;
  lesson: string;
  integration: string;
};

export type BenchmarkTierRow = {
  tier: string;
  purpose: string;
  tasks: string;
  decision: string;
};

export type BaselineRow = {
  name: string;
  whyRequired: string;
  repoStatus: string;
};

export type ArchitectureMatrixRow = {
  rank: number;
  condition: string;
  meanAcc: string;
  accMinusMajority: string;
  lossGain: string;
  labelGap: string;
  bestTask: string;
  params: string;
  read: string;
  status: EvidenceRow["status"];
};

export const repoFacts = [
  { label: "Current thesis", value: "compact structural streams need controlled tasks and causal tests" },
  { label: "Main caution", value: "relation-aware and dual-stream transformer prior art is substantial" },
  {
    label: "Latest matrix",
    value: "posttraining/RLVR pilots now run on verifier-heavy lambda, VM, DFA, cap-matching, and protocol tasks",
  },
  { label: "Best signal so far", value: "low-data BLiMP wh-island split plus formal-task pockets, all still seed-limited" },
  { label: "Main blocker", value: "showing that explicit verifier pressure makes the structural stream causally useful" },
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
      "The current evidence does not justify claims about general language modeling superiority or broad architectural novelty. The live claim is narrower: bottlenecked structural streams may be useful when trained against real structural invariants and compared against relation-aware and dual-stream baselines.",
  },
  {
    title: "The research standard",
    icon: SearchCheck,
    body:
      "A result counts only if it survives iso-parameter, relative-bias, DeBERTa-style, and relational-stream baselines, plus phase-only/bias-only ablations, multiple seeds, hard negatives, and causal representation probes. Microvalidations are instrumentation; medium literature tasks are the main evidence.",
  },
];

export const priorArtRows: PriorArtRow[] = [
  {
    family: "Relation-aware attention",
    representativeWork: "Shaw et al., Transformer-XL, ALiBi, RoPE/TAPE-style position priors",
    lesson:
      "Pairwise biases and relative addressing are established tools. A scalar relation matrix is not novel by itself.",
    integration:
      "Standard ALiBi and relative-bias conditions are mandatory comparators before claiming structural-stream gains.",
  },
  {
    family: "Disentangled streams",
    representativeWork: "DeBERTa, Abstractor, Dual Attention Transformer",
    lesson:
      "Separating object/content information from relation information is already an active architecture family.",
    integration:
      "The repo now includes DeBERTa-lite and relational-stream-lite comparators; stronger Abstractor/DAT parity remains on the implementation queue.",
  },
  {
    family: "Graph transformer structure",
    representativeWork: "Graphormer, SAN, GRPE, GraphGPS",
    lesson:
      "Graph-relative distances, spectral/Laplacian coordinates, and structural attention biases are standard for graph inputs.",
    integration:
      "When ground-truth graph structure is available, compare phase initialization or learned phase against graph-bias baselines.",
  },
  {
    family: "Program and proof structure",
    representativeWork: "GraphCodeBERT, ContraCode, S4Eq, EquiBench, LeanDojo/LeanProgress",
    lesson:
      "Semantic-preserving transformations and proof-state supervision are the right pressure for structural invariance.",
    integration:
      "Cap matching, unification, algebraic protocol closure, code-equivalence probes, and LeanProgress staging are the formal branch.",
  },
  {
    family: "Representation geometry",
    representativeWork: "nonlinear feature geometry, concept polytopes, manifold/phase geometry, intrinsic dimension work",
    lesson:
      "Useful features need not be sparse scalar axes; they can be circular, polytope-like, dense, or manifold-valued.",
    integration:
      "Interpretability now includes rank, eigenspectrum, CKA, sparse probes, archetype/convex probes, and causal patching.",
  },
  {
    family: "Intrinsic interpretability",
    representativeWork: "Sparse CLIP, concept bottlenecks, gated SAEs, ReFT, transcoders",
    lesson:
      "Interpretability can be trained in, but must be checked against performance and causal faithfulness.",
    integration:
      "Phase-contrastive training hooks and sparse/geometry diagnostics are treated as first-class conditions, not post-hoc decoration.",
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
    name: "standard_alibi",
    mechanism: "Ordinary transformer with an ALiBi causal distance prior.",
    whatItTests: "Whether gains are just better relative addressing or length/distance bias.",
  },
  {
    name: "standard_deberta_lite",
    mechanism: "Ordinary transformer with compact disentangled relative content/position attention.",
    whatItTests: "Whether a known content/position split explains the same signal.",
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
    name: "relational_stream_lite",
    mechanism: "A learned relation stream with a bilinear relation kernel and dedicated structural head, without projected phase as token content.",
    whatItTests: "Whether a simpler Abstractor/DAT-like relational pathway captures the same benefit.",
  },
  {
    name: "*_phase_contrastive",
    mechanism: "Condition suffix that adds supervised contrastive pressure to phase states on structural labels.",
    whatItTests: "Whether the structural stream needs explicit invariance pressure rather than architecture alone.",
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

export const architectureMatrixRunFacts = [
  { label: "Run", value: "architecture_matrix_smoke_2026_04_28_nextop" },
  { label: "Scope", value: "20 registered conditions x 3 tasks x 1 seed" },
  { label: "Tasks", value: "unification_depth4, listops_depth4, equibench_oj_va" },
  { label: "Scale", value: "tiny local CPU health check; not a converged benchmark" },
];

export const architectureMatrixTakeaways = [
  "All registered variants instantiate, train, emit metrics, and summarize cleanly.",
  "DeBERTa-lite baselines lead the tiny aggregate, so content/position disentanglement is a serious comparator.",
  "The static phase/full/complex/structural-head cluster ties tightly; this is implementation coverage, not a win.",
  "Bias-only tracks the inert control closely, so additive relation bias is not isolated as the useful mechanism here.",
  "phase_dynamic_qk_film is worth keeping because it produced the only positive unification label gap in the smoke.",
];

export const architectureMatrixRows: ArchitectureMatrixRow[] = [
  {
    rank: 1,
    condition: "standard_iso_deberta_lite",
    meanAcc: "59.03%",
    accMinusMajority: "+9.03",
    lossGain: "+0.0180",
    labelGap: "-0.1665",
    bestTask: "unification_depth4",
    params: "474,656",
    read: "Top aggregate; this keeps DeBERTa-style split as a required baseline.",
    status: "mixed",
  },
  {
    rank: 2,
    condition: "standard_deberta_lite",
    meanAcc: "57.64%",
    accMinusMajority: "+7.64",
    lossGain: "+0.0094",
    labelGap: "-0.1928",
    bestTask: "equibench_oj_va",
    params: "286,976",
    read: "Strong non-resonance comparator, especially on the tiny EquiBench slice.",
    status: "mixed",
  },
  {
    rank: 3,
    condition: "structural_heads_1_normalized",
    meanAcc: "56.94%",
    accMinusMajority: "+6.94",
    lossGain: "+0.0097",
    labelGap: "-0.1384",
    bestTask: "unification_depth4",
    params: "320,837",
    read: "Best structural-family aggregate, tied with the static phase cluster.",
    status: "early",
  },
  {
    rank: 4,
    condition: "phase_stream_only_normalized_phase_contrastive",
    meanAcc: "56.94%",
    accMinusMajority: "+6.94",
    lossGain: "+0.0097",
    labelGap: "-0.1387",
    bestTask: "unification_depth4",
    params: "320,836",
    read: "Contrastive hook runs, but does not separate from phase-only at this scale.",
    status: "early",
  },
  {
    rank: 5,
    condition: "phase_stream_only_normalized",
    meanAcc: "56.94%",
    accMinusMajority: "+6.94",
    lossGain: "+0.0097",
    labelGap: "-0.1387",
    bestTask: "unification_depth4",
    params: "320,836",
    read: "Keeps the separate structural stream viable without relying on additive bias.",
    status: "early",
  },
  {
    rank: 6,
    condition: "complex_directional_normalized",
    meanAcc: "56.94%",
    accMinusMajority: "+6.94",
    lossGain: "+0.0097",
    labelGap: "-0.1386",
    bestTask: "unification_depth4",
    params: "320,901",
    read: "Directional kernel is implemented, but not yet differentiated by the tiny tasks.",
    status: "early",
  },
  {
    rank: 7,
    condition: "resonance_full_normalized_phase_contrastive",
    meanAcc: "56.94%",
    accMinusMajority: "+6.94",
    lossGain: "+0.0097",
    labelGap: "-0.1386",
    bestTask: "unification_depth4",
    params: "320,836",
    read: "Full active architecture plus contrastive pressure ties the static phase cluster.",
    status: "early",
  },
  {
    rank: 8,
    condition: "resonance_full_normalized",
    meanAcc: "56.94%",
    accMinusMajority: "+6.94",
    lossGain: "+0.0097",
    labelGap: "-0.1386",
    bestTask: "unification_depth4",
    params: "320,836",
    read: "Original active condition runs cleanly, but is not uniquely favored.",
    status: "early",
  },
  {
    rank: 9,
    condition: "standard_alibi",
    meanAcc: "55.56%",
    accMinusMajority: "+5.56",
    lossGain: "+0.0093",
    labelGap: "-0.1588",
    bestTask: "equibench_oj_va",
    params: "188,704",
    read: "Distance-bias baseline remains competitive on this small regime.",
    status: "mixed",
  },
  {
    rank: 10,
    condition: "standard",
    meanAcc: "55.56%",
    accMinusMajority: "+5.56",
    lossGain: "+0.0092",
    labelGap: "-0.1566",
    bestTask: "equibench_oj_va",
    params: "188,704",
    read: "Plain baseline is not far behind; larger tasks and seeds are required.",
    status: "mixed",
  },
  {
    rank: 11,
    condition: "phase_dynamic_qk_film_normalized",
    meanAcc: "55.56%",
    accMinusMajority: "+5.56",
    lossGain: "+0.0159",
    labelGap: "-0.0949",
    bestTask: "unification_depth4",
    params: "329,316",
    read: "Middle aggregate, but the only positive unification label gap; keep in the medium matrix.",
    status: "early",
  },
  {
    rank: 12,
    condition: "phase_dynamic_attn_normalized",
    meanAcc: "55.56%",
    accMinusMajority: "+5.56",
    lossGain: "+0.0130",
    labelGap: "-0.2038",
    bestTask: "equibench_oj_va",
    params: "329,380",
    read: "Tied the top tiny EquiBench accuracy but was weak on ListOps.",
    status: "early",
  },
  {
    rank: 13,
    condition: "relational_stream_lite_normalized",
    meanAcc: "54.86%",
    accMinusMajority: "+4.86",
    lossGain: "+0.0157",
    labelGap: "-0.2178",
    bestTask: "unification_depth4",
    params: "327,141",
    read: "Abstractor/DAT-style comparator is live; stronger parity remains a priority.",
    status: "early",
  },
  {
    rank: 14,
    condition: "phase_qk_film_normalized",
    meanAcc: "54.86%",
    accMinusMajority: "+4.86",
    lossGain: "+0.0130",
    labelGap: "-0.1395",
    bestTask: "unification_depth4",
    params: "325,060",
    read: "Static Q/K modulation is viable but less interesting than dynamic Q/K modulation so far.",
    status: "early",
  },
  {
    rank: 15,
    condition: "bias_only_normalized",
    meanAcc: "54.86%",
    accMinusMajority: "+4.86",
    lossGain: "+0.0068",
    labelGap: "-0.1897",
    bestTask: "unification_depth4",
    params: "320,836",
    read: "Tracks the inert control closely; no isolated additive-bias win.",
    status: "mixed",
  },
  {
    rank: 16,
    condition: "resonance_inert_normalized",
    meanAcc: "54.86%",
    accMinusMajority: "+4.86",
    lossGain: "+0.0068",
    labelGap: "-0.1897",
    bestTask: "unification_depth4",
    params: "320,836",
    read: "Control is doing its job by catching plumbing and majority-baseline artifacts.",
    status: "mixed",
  },
  {
    rank: 17,
    condition: "standard_iso_alibi",
    meanAcc: "52.78%",
    accMinusMajority: "+2.78",
    lossGain: "+0.0164",
    labelGap: "-0.1846",
    bestTask: "unification_depth4",
    params: "314,964",
    read: "Widened ALiBi is not favored in this smoke despite decent loss gain.",
    status: "mixed",
  },
  {
    rank: 18,
    condition: "standard_iso",
    meanAcc: "50.00%",
    accMinusMajority: "+0.00",
    lossGain: "+0.0134",
    labelGap: "-0.1916",
    bestTask: "listops_depth4",
    params: "314,964",
    read: "Capacity matching alone is not a sufficient explanation in this tiny run.",
    status: "negative",
  },
  {
    rank: 19,
    condition: "resonance_dynamic_mlp_normalized",
    meanAcc: "41.67%",
    accMinusMajority: "-8.33",
    lossGain: "+0.0094",
    labelGap: "-0.1250",
    bestTask: "unification_depth4",
    params: "325,092",
    read: "Dynamic MLP update underperforms; deprioritize unless larger runs reverse it.",
    status: "negative",
  },
  {
    rank: 20,
    condition: "phase_dynamic_mlp_normalized",
    meanAcc: "41.67%",
    accMinusMajority: "-8.33",
    lossGain: "+0.0094",
    labelGap: "-0.1250",
    bestTask: "unification_depth4",
    params: "325,092",
    read: "Same dynamic-MLP failure mode as the full resonance variant.",
    status: "negative",
  },
];

export const evidenceRows: EvidenceRow[] = [
  {
    experiment: "Structural posttraining pilot",
    task: "Lambda traces, VM traces, DFA equivalence, cap matching, algebraic protocol closure",
    standard: "standard_alibi under SFT + DPO/RLVR",
    bestResonant: "phase_dynamic_qk_film_alibi and relation_value_qk_film_alibi are running",
    interpretation:
      "This is the current main branch. The question is whether verifiable rewards and trace/process tasks make the structural stream useful, not whether raw web pretraining wins immediately.",
    status: "running",
  },
  {
    experiment: "Architecture matrix smoke",
    task: "Unification + ListOps + EquiBench tiny matrix",
    standard: "55.6% mean acc",
    bestResonant: "56.9% structural/phase cluster",
    interpretation:
      "Twenty variants now run end-to-end. DeBERTa-lite baselines lead the tiny aggregate, so the result is a baseline warning and implementation milestone, not an architecture win.",
    status: "mixed",
  },
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
    name: "Lambda normal-form and trace tasks",
    source: "Local simply typed lambda-calculus generator plus exact beta reducer",
    signalTarget: "Behavioral equivalence by beta normal form, one-step validity, and whole-trace reduction validity",
    whyItMatters:
      "This is closest to the original idea: train against equivalence classes of programs and the proof/reduction paths connecting them.",
    currentStatus: "Implemented, smoke-tested, and included in posttraining/RLVR pilots.",
  },
  {
    name: "Stack VM trace and equivalence",
    source: "Local executable stack-machine interpreter",
    signalTarget: "Program execution traces and same-final-state behavioral equivalence",
    whyItMatters:
      "Adds a non-lambda machine semantics substrate with exact outcome and process rewards.",
    currentStatus: "Implemented, smoke-tested, and included in posttraining/RLVR pilots.",
  },
  {
    name: "DFA language equivalence",
    source: "Local product-automaton verifier",
    signalTarget: "Accepted-language equivalence under state renaming and transition perturbation",
    whyItMatters:
      "A standard behavioral-equivalence problem with no dependence on lambda syntax or natural language.",
    currentStatus: "Implemented and smoke-tested; added to the posttraining runner for future launches.",
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
    name: "ListOps-style tree evaluation",
    source: "Local operator-tree generator",
    signalTarget: "Nested expression parsing and modulo operator evaluation",
    whyItMatters:
      "It adds a parse-and-compose canary distinct from bracket matching and algebraic unification.",
    currentStatus: "Implemented in the structural probe runner.",
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
    name: "SCAN / ListOps / Dyck",
    source: "Public formal-language and algorithmic-reasoning tasks",
    signalTarget: "Stack, parse-tree, and compositional sequence structure",
    whyItMatters:
      "Cheap canaries for whether structural instrumentation works; not enough for headline claims.",
    currentStatus: "Dyck and local ListOps-style probes implemented; public ListOps/SCAN staging remains open.",
  },
  {
    name: "ListOps-style parse evaluation",
    source: "Local generator following the ListOps operator-tree pattern",
    signalTarget: "Tree evaluation and nested compositional parsing",
    whyItMatters:
      "Complements Dyck: bracket matching is not the same as evaluating a nested operator tree.",
    currentStatus: "Implemented as a local structural probe; public ListOps staging remains useful later.",
  },
  {
    name: "CFQ",
    source: "Compositional Freebase Questions / MCD splits",
    signalTarget: "Compositional semantic parsing under compound divergence",
    whyItMatters:
      "A medium-scale literature task where random-split success is not enough.",
    currentStatus: "Downloaded into the dataset lake and converted to generic probes.",
  },
  {
    name: "EquiBench / CETBench",
    source: "Recent program-equivalence benchmarks",
    signalTarget: "Semantic equivalence under controlled program transformations",
    whyItMatters:
      "Closest standardized test for whether structural invariance survives surface variation in code.",
    currentStatus: "EquiBench is staged and converted into tiny generic probes; CETBench remains pending.",
  },
  {
    name: "LeanProgress / LeanDojo",
    source: "Lean proof-state and proof-progress ecosystems",
    signalTarget: "Proof-state topology, tactic/progress structure, theorem dependency",
    whyItMatters:
      "Best serious formal-reasoning target once infrastructure is ready.",
    currentStatus: "Not staged yet; prioritized after current syntactic/code probes are stable.",
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

export const benchmarkTiers: BenchmarkTierRow[] = [
  {
    tier: "Tier A: canaries",
    purpose: "Catch broken adapters, saturated tasks, and shortcut-only regimes.",
    tasks: "Dyck, ListOps/SCAN, Linzen, BLiMP slices, MSGS/HANS, cap matching, unification.",
    decision: "Do not publish from these alone; use them to decide what is safe to scale.",
  },
  {
    tier: "Tier B: main evidence",
    purpose: "Run matched 10M-30M comparisons on tasks where structure is necessary.",
    tasks: "SLOG, COGS/CFQ, EquiBench/CETBench, local formal equivalence, VM traces, DFA equivalence, LeanProgress.",
    decision: "Proceed if gains survive baselines and phase/residual causal tests.",
  },
  {
    tier: "Tier C: transfer",
    purpose: "Ask whether synthetic or formal structural pressure transfers to language.",
    tasks: "BabyLM/TinyStories pretraining, then BLiMP, MSGS, HANS, Linzen, SLOG/COGS evals.",
    decision: "Scale only if structural gains transfer without collapsing ordinary LM behavior.",
  },
];

export const requiredBaselines: BaselineRow[] = [
  {
    name: "Vanilla transformer",
    whyRequired: "Establishes the basic learnability and loss trajectory for the task.",
    repoStatus: "Implemented as standard.",
  },
  {
    name: "Iso-parameter transformer",
    whyRequired: "Controls for the extra capacity introduced by phase embeddings and adapters.",
    repoStatus: "Implemented as standard_iso in structural probes.",
  },
  {
    name: "ALiBi / relative bias",
    whyRequired: "Controls for the possibility that the effect is just a stronger distance/addressing prior.",
    repoStatus: "Implemented as standard_alibi.",
  },
  {
    name: "DeBERTa-style split",
    whyRequired: "Controls against existing content/position disentanglement explaining the result.",
    repoStatus: "Implemented as standard_deberta_lite.",
  },
  {
    name: "Relational-stream comparator",
    whyRequired: "Controls against Abstractor/DAT-style explicit relation pathways.",
    repoStatus: "Implemented as relational_stream_lite; stronger parity version still needed.",
  },
  {
    name: "Graph structural bias",
    whyRequired: "Required whenever the input graph is known and Laplacian/shortest-path structure is available.",
    repoStatus: "Planned for graph-initialized phase and graph-bias comparisons.",
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
      "uv run python resonance/structural_task_probe.py --task dyck --conditions standard,standard_alibi,standard_deberta_lite,phase_stream_only_normalized --device cpu --epochs 1 --train_examples 128 --val_examples 128 --output_dir resonance/outputs/reviewer_probe",
  },
  {
    title: "Run the architecture matrix smoke",
    command: "DEVICE=cpu SEEDS=901 EPOCHS=1 bash scripts/run_architecture_matrix_smoke.sh",
  },
  {
    title: "Run cap matching",
    command:
      "uv run python resonance/structural_task_probe.py --task cap_matching --train_examples 4096 --val_examples 1024 --conditions standard,standard_iso,standard_alibi,standard_deberta_lite,phase_stream_only_normalized_phase_contrastive,relational_stream_lite --output_dir resonance/outputs/cap_matching_reviewer",
  },
  {
    title: "Run literature medium suite",
    command: "DEVICE=mps bash scripts/run_literature_medium_suite.sh",
  },
  {
    title: "Rebuild the lab notebook",
    command: "uv run python resonance/build_lab_notebook.py --root resonance/outputs --output_dir resonance/outputs/lab_notebook_hyper_live",
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
  "Claims should be stated in ordinary ML terms: baselines, objectives, benchmark splits, interventions, and reproducible measurements.",
];

export const capacityCaution = [
  "The 128d/4-layer persvati unification result reverses the 64d/2-layer ranking: Q/K-conditioned variants rise to the top while DeBERTa-lite baselines fall.",
  "This could mean structural interfaces need capacity to work, or it could be a capacity artifact, seed variance, or task-specific overfitting.",
  "Param counts are not matched across conditions: standard_iso_deberta_lite has +31% params over standard yet underperforms, which argues against a pure 'more params = better' story.",
  "Still, only 2 seeds and 1 task are complete at 128d. The pattern must replicate on cap_matching and algebraic_protocol before we treat it as real.",
  "Language modeling results (BabyLM, TinyStories) will provide an out-of-domain check: if structural variants win on LM too, the effect is less likely to be a unification-specific fluke.",
];

export const signalMatrixRows: ArchitectureMatrixRow[] = [
  {
    rank: 1,
    condition: "standard_iso_deberta_lite",
    meanAcc: "77.81%",
    accMinusMajority: "+27.81",
    lossGain: "+0.2209",
    labelGap: "+0.5495",
    bestTask: "algebraic_protocol_depth4",
    params: "585,120",
    read: "Still leads the small-model aggregate, but fell to bottom at 128d on unification.",
    status: "mixed",
  },
  {
    rank: 2,
    condition: "standard",
    meanAcc: "77.40%",
    accMinusMajority: "+27.40",
    lossGain: "+0.2091",
    labelGap: "+0.6023",
    bestTask: "algebraic_protocol_depth4",
    params: "378,112",
    read: "Strong baseline; mid-pack at 128d unification, which is a warning for capacity-artifact claims.",
    status: "mixed",
  },
  {
    rank: 3,
    condition: "phase_dynamic_qk_film_normalized",
    meanAcc: "77.29%",
    accMinusMajority: "+27.29",
    lossGain: "+0.2109",
    labelGap: "+0.5799",
    bestTask: "algebraic_protocol_depth4",
    params: "536,712",
    read: "Only structural variant that consistently competes with strong baselines on small model.",
    status: "early",
  },
  {
    rank: 4,
    condition: "resonance_full_normalized",
    meanAcc: "76.75%",
    accMinusMajority: "+26.75",
    lossGain: "+0.2026",
    labelGap: "+0.5017",
    bestTask: "algebraic_protocol_depth4",
    params: "511,304",
    read: "Tracks phase_stream_only almost exactly; static bias readout is not the active ingredient.",
    status: "mixed",
  },
  {
    rank: 5,
    condition: "phase_stream_only_normalized",
    meanAcc: "76.66%",
    accMinusMajority: "+26.66",
    lossGain: "+0.2026",
    labelGap: "+0.5056",
    bestTask: "algebraic_protocol_depth4",
    params: "511,304",
    read: "Nearly identical to resonance_full; the learned stream matters more than the bias.",
    status: "early",
  },
  {
    rank: 6,
    condition: "harmonic_cosine_normalized",
    meanAcc: "76.63%",
    accMinusMajority: "+26.63",
    lossGain: "+0.2033",
    labelGap: "+0.4949",
    bestTask: "algebraic_protocol_depth4",
    params: "511,308",
    read: "Multi-harmonic kernel does not separate from plain phase_stream_only at this scale.",
    status: "early",
  },
  {
    rank: 7,
    condition: "harmonic_relation_value_normalized",
    meanAcc: "76.59%",
    accMinusMajority: "+26.59",
    lossGain: "+0.2047",
    labelGap: "+0.4769",
    bestTask: "algebraic_protocol_depth4",
    params: "519,502",
    read: "Relation-value mixing plus harmonic kernel is viable but not a clear win yet.",
    status: "early",
  },
  {
    rank: 8,
    condition: "relation_value_mix_normalized",
    meanAcc: "76.59%",
    accMinusMajority: "+26.59",
    lossGain: "+0.2046",
    labelGap: "+0.4804",
    bestTask: "algebraic_protocol_depth4",
    params: "519,498",
    read: "Relation-value mix tracks the harmonic and full variants closely.",
    status: "early",
  },
  {
    rank: 9,
    condition: "standard_alibi",
    meanAcc: "76.52%",
    accMinusMajority: "+26.52",
    lossGain: "+0.1996",
    labelGap: "+0.5177",
    bestTask: "algebraic_protocol_depth4",
    params: "378,112",
    read: "Distance bias is competitive on small tasks; less clear at scale.",
    status: "mixed",
  },
  {
    rank: 10,
    condition: "relation_value_qk_film_normalized",
    meanAcc: "75.62%",
    accMinusMajority: "+25.62",
    lossGain: "+0.1628",
    labelGap: "+0.5332",
    bestTask: "algebraic_protocol_depth4",
    params: "536,394",
    read: "Weak on small model, but jumped to #1 at 128d/4-layer on unification. Capacity-dependent?",
    status: "early",
  },
  {
    rank: 11,
    condition: "standard_deberta_lite",
    meanAcc: "76.65%",
    accMinusMajority: "+26.65",
    lossGain: "+0.1976",
    labelGap: "+0.5979",
    bestTask: "algebraic_protocol_depth4",
    params: "443,520",
    read: "Mid-pack on small model; dropped to dead last at 128d unification.",
    status: "mixed",
  },
  {
    rank: 12,
    condition: "relational_stream_lite_normalized",
    meanAcc: "74.68%",
    accMinusMajority: "+24.68",
    lossGain: "+0.1908",
    labelGap: "+0.5869",
    bestTask: "algebraic_protocol_depth4",
    params: "522,890",
    read: "Dead last on small model; surprisingly strong at 128d. Very seed/task sensitive.",
    status: "early",
  },
];

export const persvatiUnificationRows = [
  { rank: 1, condition: "relation_value_qk_film_normalized", avgBest: "84.58%", avgFinal: "82.75%", seeds: 2, params: "1,616,788" },
  { rank: 2, condition: "standard_alibi", avgBest: "84.42%", avgFinal: "84.17%", seeds: 2, params: "1,348,352" },
  { rank: 3, condition: "relational_stream_lite_normalized", avgBest: "84.08%", avgFinal: "82.92%", seeds: 2, params: "1,505,812" },
  { rank: 4, condition: "resonance_full_normalized", avgBest: "84.00%", avgFinal: "84.00%", seeds: 2, params: "1,483,664" },
  { rank: 4, condition: "phase_dynamic_qk_film_normalized", avgBest: "84.00%", avgFinal: "83.58%", seeds: 2, params: "1,568,272" },
  { rank: 6, condition: "harmonic_relation_value_normalized", avgBest: "83.92%", avgFinal: "83.92%", seeds: 2, params: "1,549,208" },
  { rank: 7, condition: "relation_value_mix_normalized", avgBest: "83.67%", avgFinal: "83.67%", seeds: 2, params: "1,549,204" },
  { rank: 8, condition: "harmonic_cosine_normalized", avgBest: "83.50%", avgFinal: "82.67%", seeds: 2, params: "1,483,668" },
  { rank: 9, condition: "standard", avgBest: "82.75%", avgFinal: "82.42%", seeds: 2, params: "1,348,352" },
  { rank: 10, condition: "standard_iso_deberta_lite", avgBest: "82.67%", avgFinal: "79.75%", seeds: 2, params: "1,762,832" },
  { rank: 11, condition: "phase_stream_only_normalized", avgBest: "82.58%", avgFinal: "80.25%", seeds: 2, params: "1,483,664" },
  { rank: 12, condition: "standard_deberta_lite", avgBest: "82.25%", avgFinal: "81.83%", seeds: 2, params: "1,609,984" },
];

export const lmRunFacts = [
  { label: "Persvati CPU", value: "BabyLM + TinyStories, 2 seeds, 128d/3-layer" },
  { label: "Nextop CPU", value: "BabyLM + TinyStories, 3 seeds, 128d/3-layer" },
  { label: "Purpose", value: "Out-of-domain check: does phase_dynamic_qk_film help on language modeling?" },
  { label: "Status", value: "Running" },
];

export const formalDataIdeas: IconItem[] = [
  {
    title: "Proof walks",
    icon: Waypoints,
    body:
      "Generate statements, walk proof/search spaces, and train contrastively over multiple proofs of the same judgment. This directly targets the topology of deduction rather than accidental syntax.",
  },
  {
    title: "Verifier posttraining",
    icon: SearchCheck,
    body:
      "Warm-start with supervised answers, then use DPO, exact expected reward, or GRPO-style updates against deterministic verifiers for reductions, traces, automata, protocols, and equivalence.",
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
