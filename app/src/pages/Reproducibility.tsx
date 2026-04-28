import { codeArtifacts, reproducibilitySteps } from "@/data/research";

export default function Reproducibility() {
  return (
    <div className="site-shell">
      <section className="section-band page-hero">
        <p className="eyebrow">Reproducibility</p>
        <h1>The project should be runnable from a clean clone</h1>
        <p className="lead">
          The repository is being organized so a reviewer can reproduce the task generators,
          architecture sweeps, external cap-matching reference checkout, live notebooks, and plots
          without relying on local paths from this machine.
        </p>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Commands</p>
          <h2>Reviewer quick start</h2>
        </div>
        <div className="command-list">
          {reproducibilitySteps.map((step) => (
            <article className="command-row" key={step.title}>
              <strong>{step.title}</strong>
              <pre>{step.command}</pre>
            </article>
          ))}
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">Important files</p>
          <h2>Where the work lives</h2>
        </div>
        <div className="artifact-grid">
          {codeArtifacts.map((artifact) => (
            <article className="artifact-row" key={artifact.path}>
              <code>{artifact.path}</code>
              <p>{artifact.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section-band muted-band">
        <div className="split-layout">
          <div>
            <p className="eyebrow">External dependency</p>
            <h2>Cap matching reference source</h2>
            <p>
              The cap matcher reference is not assumed to live at <code>~/dev/reu_unif</code>.
              Reproduction goes through <code>scripts/setup_external_deps.sh</code>, which clones
              <code>https://github.com/emberian/reu_unif.git</code> into <code>external/reu_unif</code>
              and pins commit <code>c0da84c2d8d392031ab252e3617d43a9c9de96d0</code>. The training
              task itself is self-contained Python; the Rust source is retained as the historical
              algorithmic reference.
            </p>
          </div>
          <div className="note-panel">
            <h3>Current caveat</h3>
            <p>
              The old Rust binary does not build cleanly on modern stable Rust because it uses
              obsolete feature gates and APIs. That is documented as a reference-source issue, not a
              blocker for the Python training probes.
            </p>
          </div>
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">Compute workflow</p>
          <h2>How overnight runs are orchestrated</h2>
        </div>
        <div className="two-column">
          <article className="info-card">
            <h3>persvati</h3>
            <p>
              Remote AMD ROCm machine used for GPU-heavy second-wave sweeps and CPU remainder jobs.
              Scripts are written to run under <code>uv</code>, use ROCm-compatible device selection,
              and write structured output directories for notebook aggregation.
            </p>
          </article>
          <article className="info-card">
            <h3>nextop</h3>
            <p>
              Local Mac used for Apple MPS runs and CPU synthetic-task sweeps. The watcher script
              periodically rsyncs remote results, rebuilds the lab notebook, and refreshes plots in
              <code>resonance/outputs/lab_notebook_hyper_live</code>.
            </p>
          </article>
        </div>
      </section>
    </div>
  );
}
