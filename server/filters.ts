import type { FilterConfig } from '../src/domain/types.js';

export function applyFilter(signal: Float32Array, filter: FilterConfig): Float32Array {
  switch (filter.type) {
    case 'gain': {
      const g = filter.params.gain ?? 1;
      return Float32Array.from(signal, (x) => x * g);
    }
    case 'normalize': {
      const peak = Math.max(...signal.map((x) => Math.abs(x)), 1e-9);
      const target = filter.params.targetPeak ?? 0.9;
      return Float32Array.from(signal, (x) => (x / peak) * target);
    }
    case 'moving_average': {
      const window = Math.max(1, Math.floor(filter.params.window ?? 5));
      const out = new Float32Array(signal.length);
      for (let i = 0; i < signal.length; i++) {
        let sum = 0;
        let n = 0;
        for (let j = Math.max(0, i - window + 1); j <= i; j++) {
          sum += signal[j];
          n++;
        }
        out[i] = sum / n;
      }
      return out;
    }
    case 'high_pass': {
      const alpha = filter.params.alpha ?? 0.95;
      const out = new Float32Array(signal.length);
      out[0] = signal[0];
      for (let i = 1; i < signal.length; i++) out[i] = alpha * (out[i - 1] + signal[i] - signal[i - 1]);
      return out;
    }
    case 'low_pass': {
      const alpha = filter.params.alpha ?? 0.15;
      const out = new Float32Array(signal.length);
      out[0] = signal[0];
      for (let i = 1; i < signal.length; i++) out[i] = alpha * signal[i] + (1 - alpha) * out[i - 1];
      return out;
    }
  }
}

export function runPipeline(signal: Float32Array, pipeline: FilterConfig[]): Float32Array {
  return pipeline.reduce((acc, filter) => applyFilter(acc, filter), signal);
}
