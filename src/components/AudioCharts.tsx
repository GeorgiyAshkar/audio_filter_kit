import Plot from 'react-plotly.js';

type Props = {
  raw: Float32Array;
  processed: Float32Array;
  sampleRate: number;
  selection: [number, number] | null;
  onSelectionChange: (next: [number, number] | null) => void;
};

function fftMagnitude(signal: Float32Array, sampleRate: number): { f: number[]; mag: number[] } {
  const n = Math.min(4096, signal.length);
  if (n < 4) return { f: [], mag: [] };
  const mag: number[] = [];
  const f: number[] = [];
  for (let k = 0; k < n / 2; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      re += signal[t] * Math.cos(angle);
      im -= signal[t] * Math.sin(angle);
    }
    f.push((k * sampleRate) / n);
    mag.push(20 * Math.log10(Math.sqrt(re * re + im * im) + 1e-12));
  }
  return { f, mag };
}

export function AudioCharts({ raw, processed, sampleRate, selection, onSelectionChange }: Props) {
  const time = Array.from({ length: raw.length }, (_, i) => i / sampleRate);
  const [from, to] = selection ?? [0, Math.max(time.at(-1) ?? 0, 0.1)];
  const start = Math.max(0, Math.floor(from * sampleRate));
  const end = Math.min(raw.length, Math.ceil(to * sampleRate));
  const spec = fftMagnitude(processed.slice(start, end), sampleRate);

  return (
    <div className="charts">
      <Plot
        data={[{ x: time, y: Array.from(raw), type: 'scatter', mode: 'lines', name: 'overview' }]}
        layout={{ height: 160, margin: { l: 40, r: 10, t: 20, b: 30 }, xaxis: { rangeslider: { visible: true } } }}
        onRelayout={(e) => {
          const x0 = e['xaxis.range[0]'];
          const x1 = e['xaxis.range[1]'];
          if (typeof x0 === 'number' && typeof x1 === 'number') onSelectionChange([x0, x1]);
        }}
        config={{ displayModeBar: false }}
      />
      <Plot
        data={[
          { x: time, y: Array.from(raw), type: 'scatter', mode: 'lines', name: 'Raw' },
          { x: time, y: Array.from(processed), type: 'scatter', mode: 'lines', name: 'Processed' }
        ]}
        layout={{ height: 300, margin: { l: 40, r: 10, t: 20, b: 30 }, xaxis: { range: [from, to] } }}
        config={{ responsive: true }}
      />
      <Plot
        data={[{ x: spec.f, y: spec.mag, type: 'scatter', mode: 'lines', name: 'PSD' }]}
        layout={{ height: 280, margin: { l: 40, r: 10, t: 20, b: 30 }, xaxis: { title: { text: 'Hz' } }, yaxis: { title: { text: 'dB' } } }}
        config={{ responsive: true }}
      />
    </div>
  );
}
