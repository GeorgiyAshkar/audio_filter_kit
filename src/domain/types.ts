export type FilterType = 'gain' | 'normalize' | 'moving_average' | 'high_pass' | 'low_pass';

export interface FilterConfig {
  id: string;
  type: FilterType;
  name: string;
  params: Record<string, number>;
}

export interface ProcessingMetrics {
  snrDb: number;
  rms: number;
  peak: number;
  latencyMs: number;
}

export interface ProcessResponse {
  processed: number[];
  sampleRate: number;
  metrics: ProcessingMetrics;
}
