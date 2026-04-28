import {
  architectureMatrixRows,
  architectureMatrixRunFacts,
  architectureMatrixTakeaways,
  architectureSteps,
  architectureVariants,
  requiredBaselines,
} from "@/data/research";

function StatusPill({ status }: { status: string }) {
  return <span className={`status-pill ${status.replaceAll(" ", "-")}`}>{status}</span>;
}

export default function Architecture() {
  return (
    <div className="site-shell">
      <section className="section-band page-hero">
        <p className="eyebrow">Architecture</p>
        <h1>From additive resonance bias to a family of structural-stream tests</h1>
        <p className="lead">
          The original model added a phase-derived relation matrix to attention logits. That was a
          reasonable first implementation, but it is not sacred. The current program treats it as
          one member of a larger design space: static phase streams, dynamic phase states,
          phase-conditioned attention geometry, directional kernels, and dedicated structural heads.
        </p>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Core construction</p>
          <h2>The minimal model</h2>
        </div>
        <div className="step-list">
          {architectureSteps.map((step) => (
            <article className="step-row" key={step.label}>
              <strong>{step.label}</strong>
              <p>{step.detail}</p>
            </article>
          ))}
        </div>
        <div className="code-block">
          <pre>{`semantic = E_sem[token_ids]                 # [batch, seq, d_model]
phase = E_phase[token_ids]                  # [batch, seq, n_phase]
R[i,j] = mean_f cos(phase[i,f] - phase[j,f])
attention_logits = QK^T / sqrt(d_head) + w * normalize(R)
hidden = transformer_blocks(blend(semantic, project(phase)))`}</pre>
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">Ablation surface</p>
          <h2>Every condition asks a different causal question</h2>
        </div>
        <div className="table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Condition</th>
                <th>Mechanism</th>
                <th>What it tests</th>
              </tr>
            </thead>
            <tbody>
              {architectureVariants.map((variant) => (
                <tr key={variant.name}>
                  <td>
                    <code>{variant.name}</code>
                  </td>
                  <td>{variant.mechanism}</td>
                  <td>{variant.whatItTests}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Local matrix smoke</p>
          <h2>Every registered variant now runs through the same harness</h2>
          <p>
            This is a tiny implementation-health matrix, not a converged benchmark. Its value is
            that it exercises the full architectural surface and tells us which comparators and
            variants deserve real medium-scale runs.
          </p>
        </div>
        <div className="two-column">
          {architectureMatrixRunFacts.map((fact) => (
            <article className="note-panel" key={fact.label}>
              <h3>{fact.label}</h3>
              <p>{fact.value}</p>
            </article>
          ))}
        </div>
        <div className="note-panel section-note">
          <h3>Readout</h3>
          <ul className="compact-list">
            {architectureMatrixTakeaways.map((takeaway) => (
              <li key={takeaway}>{takeaway}</li>
            ))}
          </ul>
        </div>
        <div className="table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Condition</th>
                <th>Mean acc</th>
                <th>Acc-maj</th>
                <th>Loss gain</th>
                <th>Label gap</th>
                <th>Best task</th>
                <th>Status</th>
                <th>Read</th>
              </tr>
            </thead>
            <tbody>
              {architectureMatrixRows.map((row) => (
                <tr key={row.condition}>
                  <td>{row.rank}</td>
                  <td>
                    <code>{row.condition}</code>
                  </td>
                  <td>{row.meanAcc}</td>
                  <td>{row.accMinusMajority}</td>
                  <td>{row.lossGain}</td>
                  <td>{row.labelGap}</td>
                  <td>
                    <code>{row.bestTask}</code>
                  </td>
                  <td>
                    <StatusPill status={row.status} />
                  </td>
                  <td>{row.read}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">Baseline discipline</p>
          <h2>Comparators required by the literature review</h2>
        </div>
        <div className="table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Baseline</th>
                <th>Why it is required</th>
                <th>Repo status</th>
              </tr>
            </thead>
            <tbody>
              {requiredBaselines.map((baseline) => (
                <tr key={baseline.name}>
                  <td>{baseline.name}</td>
                  <td>{baseline.whyRequired}</td>
                  <td>{baseline.repoStatus}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-band">
        <div className="split-layout">
          <div>
            <p className="eyebrow">Design correction</p>
            <h2>The phrase "resonance bias is causal" means one specific thing</h2>
            <p>
              It does not mean the broader structural-prior hypothesis is wrong or right. It means
              the additive bias path itself changes the answer after controlling for the extra phase
              representation and extra parameters. Early BLiMP results do not yet show that cleanly:
              phase-only can match or beat the full model, while bias-only is weaker.
            </p>
          </div>
          <div className="note-panel">
            <h3>Current architectural bet</h3>
            <p>
              Treat phase as a structural state, not merely as a scalar logit bonus. The most
              interesting next variants are dynamic phase updates, Q/K modulation, directional
              kernels for asymmetric relations, and structural heads that can specialize without
              forcing every head through the same interface.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
