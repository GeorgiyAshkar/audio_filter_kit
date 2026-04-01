import { useMemo, useState } from 'react';
import { WaveformView } from './components/WaveformView';
import { filterCatalog } from './filters/basicFilters';
import { decodeAudioFile, cropBySelection } from './services/audioService';
import { computeMetrics } from './services/metricsService';
import { runPipeline } from './services/pipelineService';
import type { AudioFilter, AudioTrack, SelectionRange } from './types';
import './styles.css';

function App() {
  const [raw, setRaw] = useState<AudioTrack | null>(null);
  const [reference, setReference] = useState<AudioTrack | null>(null);
  const [processed, setProcessed] = useState<Float32Array | null>(null);
  const [selection, setSelection] = useState<SelectionRange>(null);
  const [pipeline, setPipeline] = useState<AudioFilter[]>([]);
  const [selectedCatalog, setSelectedCatalog] = useState<string[]>([]);

  const metrics = useMemo(() => {
    if (!reference || !processed) return null;
    return computeMetrics(reference.samples, processed);
  }, [reference, processed]);

  const addSelected = () => {
    const next = selectedCatalog
      .map((id) => filterCatalog.find((f) => new f().id === id))
      .filter((f): f is (typeof filterCatalog)[number] => Boolean(f))
      .map((Ctor) => new Ctor());
    setPipeline((prev) => [...prev, ...next]);
  };

  const applyPipeline = () => {
    if (!raw || pipeline.length === 0) return;
    const segment = cropBySelection(raw, selection);
    const out = runPipeline(segment, raw.sampleRate, pipeline);
    setProcessed(out);
  };

  return (
    <div className="app-shell">
      <header><h1>DAS Speech Filter Explorer (Web / TypeScript)</h1></header>
      <div className="toolbar">
        <label>Загрузить noisy/raw <input type="file" accept="audio/*" onChange={async (e) => {
          const file = e.target.files?.[0]; if (!file) return; setRaw(await decodeAudioFile(file));
        }} /></label>
        <label>Загрузить reference <input type="file" accept="audio/*" onChange={async (e) => {
          const file = e.target.files?.[0]; if (!file) return; setReference(await decodeAudioFile(file));
        }} /></label>
      </div>

      <div className="meta">
        <div>Raw: {raw ? `${raw.name} | sr=${raw.sampleRate} | samples=${raw.samples.length}` : '—'}</div>
        <div>Reference: {reference ? `${reference.name} | sr=${reference.sampleRate} | samples=${reference.samples.length}` : '—'}</div>
        <div>Metrics: {metrics ? `MSE=${metrics.mse.toFixed(6)} | SNR=${metrics.snr.toFixed(2)} dB | SegSNR=${metrics.segSnr.toFixed(2)} dB` : '—'}</div>
      </div>

      <main className="content-grid">
        <section className="panel">
          <h3>Доступные фильтры</h3>
          {filterCatalog.map((Ctor) => {
            const f = new Ctor();
            return (
              <label key={f.id} className="list-row">
                <input
                  type="checkbox"
                  checked={selectedCatalog.includes(f.id)}
                  onChange={(e) => setSelectedCatalog((prev) => e.target.checked ? [...prev, f.id] : prev.filter((x) => x !== f.id))}
                />
                {f.name}
              </label>
            );
          })}
          <button onClick={addSelected}>Добавить →</button>
        </section>

        <section className="panel">
          <h3>Pipeline</h3>
          {pipeline.map((f, idx) => (
            <div key={`${f.id}-${idx}`} className="pipeline-card">
              <strong>{idx + 1}. {f.name}</strong>
              {Object.entries(f.paramSpec()).map(([name, spec]) => (
                <label key={name}>
                  {name}
                  <input
                    type="range"
                    min={spec.min}
                    max={spec.max}
                    step={spec.step}
                    value={f.params[name]}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      setPipeline((prev) => prev.map((x, i) => {
                        if (i !== idx) return x;
                        const cloned = Object.create(Object.getPrototypeOf(x)) as AudioFilter;
                        Object.assign(cloned, x, { params: { ...x.params, [name]: v } });
                        return cloned;
                      }));
                    }}
                  />
                </label>
              ))}
            </div>
          ))}
          <div className="btn-row">
            <button onClick={applyPipeline}>Применить pipeline</button>
            <button onClick={() => setPipeline([])}>Очистить</button>
          </div>
        </section>

        <section className="panel wide">
          <WaveformView
            raw={raw?.samples ?? null}
            reference={reference?.samples ?? null}
            processed={processed}
            sampleRate={raw?.sampleRate ?? 0}
            selection={selection}
            onSelectionChange={setSelection}
          />
        </section>
      </main>
    </div>
  );
}

export default App;
