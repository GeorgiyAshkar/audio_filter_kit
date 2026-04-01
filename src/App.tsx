import { useMemo, useState } from 'react';
import { AudioCharts } from './components/AudioCharts';
import { createFilter, FilterPanel } from './components/FilterPanel';
import type { FilterConfig, FilterType, ProcessingMetrics } from './domain/types';
import { decodeAudio, encodeWav, processSignal } from './services/audio';

const zeroMetrics: ProcessingMetrics = { snrDb: 0, rms: 0, peak: 0, latencyMs: 0 };

export function App() {
  const [raw, setRaw] = useState<Float32Array>(new Float32Array());
  const [processed, setProcessed] = useState<Float32Array>(new Float32Array());
  const [sampleRate, setSampleRate] = useState(44100);
  const [pipeline, setPipeline] = useState<FilterConfig[]>([]);
  const [selection, setSelection] = useState<[number, number] | null>(null);
  const [metrics, setMetrics] = useState<ProcessingMetrics>(zeroMetrics);
  const [status, setStatus] = useState('ready');

  const audioUrl = useMemo(() => {
    if (!processed.length) return null;
    return URL.createObjectURL(encodeWav(processed, sampleRate));
  }, [processed, sampleRate]);

  const onUpload = async (file?: File) => {
    if (!file) return;
    setStatus('loading...');
    const decoded = await decodeAudio(file);
    setRaw(decoded.data);
    setProcessed(decoded.data);
    setSampleRate(decoded.sampleRate);
    setSelection([0, decoded.data.length / decoded.sampleRate]);
    setStatus('loaded');
  };

  const applyPipeline = async () => {
    setStatus('processing...');
    const result = await processSignal(raw, sampleRate, pipeline);
    setProcessed(Float32Array.from(result.processed));
    setMetrics(result.metrics);
    setStatus('done');
  };

  return (
    <div className="layout">
      <header><h2>DAS Speech Filter Explorer (Web TypeScript)</h2></header>

      <div className="toolbar">
        <label className="upload">Загрузить noisy/raw <input type="file" accept="audio/*" onChange={(e) => onUpload(e.target.files?.[0])} /></label>
        <button onClick={applyPipeline} disabled={!raw.length}>Применить pipeline</button>
        <span>Status: {status}</span>
      </div>

      <div className="content">
        <FilterPanel
          pipeline={pipeline}
          onAdd={(type: FilterType) => setPipeline((prev) => [...prev, createFilter(type)])}
          onRemove={(id) => setPipeline((prev) => prev.filter((f) => f.id !== id))}
          onUpdate={(id, key, value) => setPipeline((prev) => prev.map((f) => (f.id === id ? { ...f, params: { ...f.params, [key]: value } } : f)))}
        />

        <div className="main-panel">
          {!!raw.length && (
            <AudioCharts
              raw={raw}
              processed={processed}
              sampleRate={sampleRate}
              selection={selection}
              onSelectionChange={setSelection}
            />
          )}
          <div className="metrics">
            <span>SNR: {metrics.snrDb.toFixed(2)} dB</span>
            <span>RMS: {metrics.rms.toFixed(4)}</span>
            <span>Peak: {metrics.peak.toFixed(4)}</span>
            <span>Latency: {metrics.latencyMs.toFixed(2)} ms</span>
          </div>
          {audioUrl && <audio controls src={audioUrl} />}
        </div>
      </div>
    </div>
  );
}
