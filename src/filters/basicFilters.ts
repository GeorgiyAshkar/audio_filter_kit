import type { AudioFilter, FilterParamSpec } from '../types';

abstract class BaseFilter implements AudioFilter {
  constructor(
    public id: string,
    public name: string,
    public params: Record<string, number>
  ) {}

  abstract paramSpec(): Record<string, FilterParamSpec>;

  process(samples: Float32Array): Float32Array {
    return samples;
  }
}

export class GainFilter extends BaseFilter {
  constructor() {
    super('gain', 'Gain', { gain: 1 });
  }
  paramSpec() {
    return { gain: { type: 'number', min: 0.1, max: 3, step: 0.1, value: this.params.gain } };
  }
  process(samples: Float32Array): Float32Array {
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) out[i] = samples[i] * this.params.gain;
    return out;
  }
}

export class MovingAverageFilter extends BaseFilter {
  constructor() {
    super('moving_avg', 'Moving average', { window: 5 });
  }
  paramSpec() {
    return { window: { type: 'number', min: 1, max: 101, step: 2, value: this.params.window } };
  }
  process(samples: Float32Array): Float32Array {
    const w = Math.max(1, Math.floor(this.params.window));
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      let acc = 0;
      let cnt = 0;
      for (let k = Math.max(0, i - w); k <= i; k += 1) {
        acc += samples[k];
        cnt += 1;
      }
      out[i] = acc / cnt;
    }
    return out;
  }
}

export class NoiseGateFilter extends BaseFilter {
  constructor() {
    super('noise_gate', 'VAD noise gate', { threshold: 0.02 });
  }
  paramSpec() {
    return { threshold: { type: 'number', min: 0.001, max: 0.2, step: 0.001, value: this.params.threshold } };
  }
  process(samples: Float32Array): Float32Array {
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      out[i] = Math.abs(samples[i]) < this.params.threshold ? 0 : samples[i];
    }
    return out;
  }
}

export const filterCatalog = [GainFilter, MovingAverageFilter, NoiseGateFilter];
