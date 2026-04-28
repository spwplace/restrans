import { evidenceRows } from "@/data/research";

const signalBars = [
  { label: "BLiMP wh-island n=150", standard: 71.1, resonance: 79.3 },
  { label: "BLiMP wh-island n=100", standard: 56.5, resonance: 66.0 },
  { label: "Cap matching smoke", standard: 53.1, resonance: 54.2 },
  { label: "Algebraic protocol smoke", standard: 60.4, resonance: 61.5 },
  { label: "Term unification seed 201", standard: 75.6, resonance: 79.0 },
];

function StatusPill({ status }: { status: string }) {
  return <span className={`status-pill ${status.replaceAll(" ", "-")}`}>{status}</span>;
}

export default function Experiments() {
  return (
    <div className="site-shell">
      <section className="section-band page-hero">
        <p className="eyebrow">Experiments</p>
        <h1>Current evidence is real enough to pursue, not strong enough to sell</h1>
        <p className="lead">
          The program has moved from broad claims to regime-finding. We want tasks where structure
          is necessary, labels are exact, surface shortcuts are controlled, and the model is neither
          saturated nor lost. Only after a task passes that probe does a full architecture sweep mean
          anything.
        </p>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Snapshot</p>
          <h2>Result table</h2>
        </div>
        <div className="table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Task</th>
                <th>Standard</th>
                <th>Best resonant/phase condition</th>
                <th>Interpretation</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {evidenceRows.map((row) => (
                <tr key={row.experiment}>
                  <td>{row.experiment}</td>
                  <td>{row.task}</td>
                  <td>{row.standard}</td>
                  <td>{row.bestResonant}</td>
                  <td>{row.interpretation}</td>
                  <td>
                    <StatusPill status={row.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">Core plot, in words</p>
          <h2>Accuracy deltas on early structural tasks</h2>
        </div>
        <div className="bar-list">
          {signalBars.map((row) => (
            <div className="bar-row" key={row.label}>
              <div className="bar-label">
                <strong>{row.label}</strong>
                <span>{(row.resonance - row.standard).toFixed(1)} point delta</span>
              </div>
              <div className="bar-track" aria-label={`${row.label} standard ${row.standard} resonance ${row.resonance}`}>
                <div className="bar standard-bar" style={{ width: `${row.standard}%` }}>
                  {row.standard.toFixed(1)}
                </div>
                <div className="bar resonance-bar" style={{ width: `${row.resonance}%` }}>
                  {row.resonance.toFixed(1)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="section-band muted-band">
        <div className="split-layout">
          <div>
            <p className="eyebrow">Reading the results</p>
            <h2>The important distinction is phase stream vs. attention bias</h2>
            <p>
              The strongest result so far is not "the resonance bias works." It is closer to:
              "when the task is structural and low-data, giving the model a separate low-dimensional
              structural stream can help." That is a better and more testable claim. It explains why
              phase-only can do well, and it motivates variants where phase conditions Q/K geometry
              or evolves as a state rather than merely adding a matrix to logits.
            </p>
          </div>
          <div className="note-panel">
            <h3>Current run locations</h3>
            <ul className="compact-list">
              <li>
                <code>resonance/outputs/hyper_2026_04_28</code> for first-wave sweeps.
              </li>
              <li>
                <code>resonance/outputs/hyper_second_wave_2026_04_28</code> for remote GPU jobs.
              </li>
              <li>
                <code>resonance/outputs/hyper_mps_algebraic_2026_04_28</code> for local MPS jobs.
              </li>
              <li>
                <code>resonance/outputs/lab_notebook_hyper_live</code> for aggregated plots and tables.
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
