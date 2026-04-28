import { datasetRows, formalDataIdeas, syntheticTasks } from "@/data/research";

export default function Tasks() {
  return (
    <div className="site-shell">
      <section className="section-band page-hero">
        <p className="eyebrow">Tasks and data</p>
        <h1>The dataset has to contain the structure we expect the model to use</h1>
        <p className="lead">
          A random synthetic task is not a fair test of a structural prior. The right task has a
          latent topology, exact labels, controlled surface cues, and enough difficulty that a
          standard transformer has to work. Microvalidations exist to find those regimes before
          spending compute on larger runs.
        </p>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Regime probe</p>
          <h2>A task is worth sweeping only if it passes these checks</h2>
        </div>
        <div className="three-column">
          <article className="info-card">
            <h3>Not saturated</h3>
            <p>
              If a small standard model immediately reaches near-perfect accuracy, the task is too
              easy or leaks lexical shortcuts. It will not reveal architecture signal.
            </p>
          </article>
          <article className="info-card">
            <h3>Not impossible</h3>
            <p>
              If all conditions sit at chance, the generator is too hard, labels are too noisy, or
              the training horizon is too short. This also produces no useful signal.
            </p>
          </article>
          <article className="info-card">
            <h3>Geometry-visible</h3>
            <p>
              Hidden states should separate the relevant structural labels. If accuracy rises while
              geometry is blank, the model may be exploiting a shortcut.
            </p>
          </article>
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">Synthetic generators</p>
          <h2>Controlled tasks we can instrument completely</h2>
        </div>
        <div className="table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Task</th>
                <th>Source</th>
                <th>Signal target</th>
                <th>Why it matters</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {syntheticTasks.map((task) => (
                <tr key={task.name}>
                  <td>{task.name}</td>
                  <td>{task.source}</td>
                  <td>{task.signalTarget}</td>
                  <td>{task.whyItMatters}</td>
                  <td>{task.currentStatus}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-band muted-band">
        <div className="section-heading">
          <p className="eyebrow">Dataset lake</p>
          <h2>Literature tasks for external validity</h2>
        </div>
        <div className="table-wrap">
          <table className="research-table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Source</th>
                <th>Signal target</th>
                <th>Why it matters</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {datasetRows.map((task) => (
                <tr key={task.name}>
                  <td>{task.name}</td>
                  <td>{task.source}</td>
                  <td>{task.signalTarget}</td>
                  <td>{task.whyItMatters}</td>
                  <td>{task.currentStatus}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-band">
        <div className="section-heading">
          <p className="eyebrow">Training bridge</p>
          <h2>How formal structure should meet natural language</h2>
        </div>
        <div className="three-column">
          {formalDataIdeas.map((idea) => {
            const Icon = idea.icon;
            return (
              <article className="info-card" key={idea.title}>
                <Icon aria-hidden="true" size={22} />
                <h3>{idea.title}</h3>
                <p>{idea.body}</p>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
