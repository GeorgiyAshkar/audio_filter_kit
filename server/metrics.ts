export function calcMetrics(raw: Float32Array, processed: Float32Array, latencyMs: number) {
  const rms = Math.sqrt(processed.reduce((acc, x) => acc + x * x, 0) / Math.max(1, processed.length));
  const peak = processed.reduce((acc, x) => Math.max(acc, Math.abs(x)), 0);
  const noise = processed.map((x, i) => x - (raw[i] ?? 0));
  const pSignal = raw.reduce((acc, x) => acc + x * x, 0) / Math.max(1, raw.length);
  const pNoise = noise.reduce((acc, x) => acc + x * x, 0) / Math.max(1, noise.length);
  const snrDb = 10 * Math.log10((pSignal + 1e-12) / (pNoise + 1e-12));
  return { snrDb, rms, peak, latencyMs };
}
