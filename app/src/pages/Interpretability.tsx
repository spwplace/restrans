import { interpretabilityPlan, openQuestions } from "@/data/research";

export default function Interpretability() {
  return (
    <div className="site-shell">
      <section className="section-band page-hero">
        <p className="eyebrow">Interpretability</p>
        <h1>The phase stream is interesting only if we can inspect it causally</h1>
        <p className="lead">
          The attractive possibility is intrinsic interpretability: a model whose structural
          coordinates are easier to probe than an ordinary residual stream. That is a prediction,
          not an assumption. The plan is to test it with ablations, patching, SAEs, linear probes,
          geometry metrics, and task-specific counterfactuals.
        </p>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Tooling plan</p>
          <h2>Mechanistic questions we can answer directly</h2>
        </div>
        <div className="three-column">
          {interpretabilityPlan.map((item) => {
            const Icon = item.icon;
            return (
              <article className="info-card" key={item.title}>
                <Icon aria-hidden="true" size={22} />
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </article>
            );
          })}
        </div>
      </section>

      <section className="section-band">
        <div className="split-layout">
          <div>
            <p className="eyebrow">Predictions</p>
            <h2>What would be interesting at scale</h2>
            <p>
              If the idea is working, structural labels should become easier to decode from phase
              states than from semantic residuals, phase perturbations should selectively damage
              structural judgments, and matched examples should cluster by proof/graph/syntax role
              even when their lexical content is disjoint. In larger models, the strongest version
              of the prediction is that phase becomes a compact structural scratchpad while semantic
              content remains high-dimensional and superposed.
            </p>
            <p>
              The Sparse CLIP and Minkowski-representation discussions are relevant because they
              warn against one-size-fits-all interpretability. We should not assume the right probe
              is always sparse and linear. We should compare sparse features, dense geometry,
              convex-region/archetype structure, and task-local causal interventions.
            </p>
          </div>
          <div className="note-panel">
            <h3>Minimum viable interp bundle</h3>
            <ul className="compact-list">
              <li>Save phase, semantic, attention, and final residual activations.</li>
              <li>Run stream ablations on every task result table.</li>
              <li>Patch positive/negative pairs with shared words but changed structure.</li>
              <li>Train probes for task labels and nuisance lexical labels.</li>
              <li>Plot rank, CKA, perturbation curves, and cluster purity by condition.</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Open questions</p>
          <h2>What the next experiments need to decide</h2>
        </div>
        <div className="two-column">
          {openQuestions.map((question) => {
            const Icon = question.icon;
            return (
              <article className="info-card" key={question.title}>
                <Icon aria-hidden="true" size={22} />
                <h3>{question.title}</h3>
                <p>{question.body}</p>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
