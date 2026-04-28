import { Link } from "react-router";
import { ArrowRight, ExternalLink, Github, ShieldCheck } from "lucide-react";
import { repoFacts, pageSummaries, priorArtRows, relationAwarePriorArt, thesisCards } from "@/data/research";

export default function Home() {
  return (
    <div className="site-shell">
      <section className="intro-band">
        <div className="content-grid">
          <div className="intro-copy">
            <p className="eyebrow">Research dossier · updated April 28, 2026</p>
            <h1>Resonance Transformers as a structural-prior research program</h1>
            <p className="lead">
              This project studies whether a dedicated phase-like stream can help transformer
              models represent topology-bearing structure: binding, graph relations, proof state,
              unification constraints, protocol traces, and controlled natural-language syntax.
            </p>
            <p>
              The current evidence is promising but narrow. The strongest early signal is on a
              low-data BLiMP wh-island regime, and it points more toward the usefulness of a
              separate phase/structural stream than toward the original additive attention-bias
              mechanism. The project is now organized around falsifiable ablations, better tasks,
              and interpretability tools that can say what the phase stream actually encodes.
            </p>
            <div className="button-row">
              <Link className="action-button" to="/experiments">
                Current evidence
                <ArrowRight aria-hidden="true" size={16} />
              </Link>
              <a className="quiet-button" href="https://github.com/spwplace/restrans" target="_blank" rel="noreferrer">
                <Github aria-hidden="true" size={16} />
                Repository
                <ExternalLink aria-hidden="true" size={14} />
              </a>
            </div>
          </div>

          <div className="fact-panel" aria-label="Project facts">
            {repoFacts.map((fact) => (
              <div className="fact-row" key={fact.label}>
                <span>{fact.label}</span>
                <strong>{fact.value}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">What the project is trying to prove</p>
          <h2>Precise enough to be wrong</h2>
        </div>
        <div className="three-column">
          {thesisCards.map((card) => {
            const Icon = card.icon;
            return (
              <article className="info-card" key={card.title}>
                <Icon aria-hidden="true" size={22} />
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </article>
            );
          })}
        </div>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Read this first</p>
          <h2>How to navigate the dossier</h2>
        </div>
        <div className="page-grid">
          {pageSummaries.map((page) => {
            const Icon = page.icon;
            return (
              <Link className="page-tile" to={page.href} key={page.title}>
                <Icon aria-hidden="true" size={21} />
                <span>{page.title}</span>
                <p>{page.body}</p>
              </Link>
            );
          })}
        </div>
      </section>

      <section className="section-band">
        <div className="split-layout">
          <div>
            <p className="eyebrow">Prior art first</p>
            <h2>The research starts inside an occupied design space</h2>
            <p>
              Relation-aware attention, graph transformers, content/position disentanglement,
              program contrast, proof-state learning, and intrinsic interpretability already give
              us the right comparison class. The current project is valuable only if it shows a
              compact structural stream doing useful, inspectable work beyond those baselines.
            </p>
          </div>
          <div className="check-list">
            {relationAwarePriorArt.map((item) => (
              <div className="check-row" key={item}>
                <ShieldCheck aria-hidden="true" size={18} />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">Literature integration</p>
          <h2>What gets folded into the next experiments</h2>
        </div>
        <div className="table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Family</th>
                <th>Representative work</th>
                <th>Lesson</th>
                <th>Integration</th>
              </tr>
            </thead>
            <tbody>
              {priorArtRows.map((row) => (
                <tr key={row.family}>
                  <td>{row.family}</td>
                  <td>{row.representativeWork}</td>
                  <td>{row.lesson}</td>
                  <td>{row.integration}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
