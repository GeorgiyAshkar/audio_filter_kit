export type AudioTrack = {
  name: string;
  sampleRate: number;
  samples: Float32Array;
};

export type FilterParamSpec = {
  type: 'number';
  min: number;
  max: number;
  step: number;
  value: number;
};

export interface AudioFilter {
  id: string;
  name: string;
  params: Record<string, number>;
  paramSpec(): Record<string, FilterParamSpec>;
  process(samples: Float32Array, sampleRate: number): Float32Array;
}

export type SelectionRange = { start: number; end: number } | null;
