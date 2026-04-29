import { architectureMatrixTakeaways, capacityCaution, evidenceRows, lmRunFacts, persvatiUnificationRows, signalMatrixRows } from "@/data/research";

const signalBars = [
  { label: "Matrix smoke: standard vs structural cluster", standard: 55.6, resonance: 56.9 },
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
          <p className="eyebrow">Signal matrix (nextop MPS)</p>
          <h2>Small-model 3-task aggregate: 64d / 2 layers / 3 seeds / 4 epochs</h2>
          <p>
            This is the first focused matrix after the smoke test. It skips weak canaries (ListOps)
            and spends compute on tasks where structural interfaces might matter: unification depth 5,
            cap matching depth 4, and algebraic protocol depth 4.
          </p>
        </div>
        <div className="table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Condition</th>
                <th>Mean acc</th>
                <th>Best task</th>
                <th>Read</th>
              </tr>
            </thead>
            <tbody>
              {signalMatrixRows.slice(0, 8).map((row) => (
                <tr key={row.condition}>
                  <td>{row.rank}</td>
                  <td>
                    <code>{row.condition}</code>
                  </td>
                  <td>{row.meanAcc}</td>
                  <td>
                    <code>{row.bestTask}</code>
                  </td>
                  <td>{row.read}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note-panel section-note">
          <h3>Small-model takeaway</h3>
          <ul className="compact-list">
            {architectureMatrixTakeaways.map((takeaway) => (
              <li key={takeaway}>{takeaway}</li>
            ))}
          </ul>
        </div>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Medium-scale persvati (ROCm)</p>
          <h2>Unification depth 5 at 128d / 4 layers / 2 seeds / 5 epochs</h2>
          <p>
            The ranking flips completely at larger scale. Q/K-conditioned structural variants rise
            to the top, while DeBERTa-lite baselines drop to the bottom. This is either a real
            capacity-dependent effect or an artifact of seed variance and param mismatch.
          </p>
        </div>
        <div className="table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Condition</th>
                <th>Avg best val acc</th>
                <th>Avg final</th>
                <th>Params</th>
              </tr>
            </thead>
            <tbody>
              {persvatiUnificationRows.map((row) => (
                <tr key={row.condition}>
                  <td>{row.rank}</td>
                  <td>
                    <code>{row.condition}</code>
                  </td>
                  <td>{row.avgBest}</td>
                  <td>{row.avgFinal}</td>
                  <td>{row.params}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note-panel section-note">
          <h3>Capacity caution</h3>
          <ul className="compact-list">
            {capacityCaution.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">Language modeling</p>
          <h2>CPU LM runs on BabyLM and TinyStories</h2>
          <p>
            Synthetic tasks are necessary but not sufficient. If structural streams only help on
            hand-designed unification tasks, the effect is not general. Language modeling provides
            an out-of-domain check where structure is implicit and surface statistics are strong.
          </p>
        </div>
        <div className="two-column">
          {lmRunFacts.map((fact) => (
            <article className="note-panel" key={fact.label}>
              <h3>{fact.label}</h3>
              <p>{fact.value}</p>
            </article>
          ))}
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
                <code>resonance/outputs/signal_matrix_nextop_2026_04_28_mps</code> for local MPS signal matrix.
              </li>
              <li>
                <code>resonance/outputs/signal_matrix_persvati_2026_04_28_rocm_v2</code> for remote GPU medium matrix.
              </li>
              <li>
                <code>resonance/outputs/cpu_lm_focused_2026_04_28</code> for persvati CPU LM runs.
              </li>
              <li>
                <code>resonance/outputs/cpu_lm_nextop_tinystories_2026_04_28</code> for local CPU TinyStories.
              </li>
              <li>
                <code>resonance/outputs/cpu_lm_nextop_babylm_2026_04_28</code> for local CPU BabyLM.
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
