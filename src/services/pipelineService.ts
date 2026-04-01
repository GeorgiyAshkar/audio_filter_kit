import type { AudioFilter } from '../types';

export function runPipeline(input: Float32Array, sampleRate: number, pipeline: AudioFilter[]): Float32Array {
  return pipeline.reduce((acc, filter) => filter.process(acc, sampleRate), input);
}
