export type Metrics = { mse: number; snr: number; segSnr: number };

export function computeMetrics(ref: Float32Array, est: Float32Array): Metrics {
  const n = Math.min(ref.length, est.length);
  if (n === 0) return { mse: Infinity, snr: -120, segSnr: -120 };

  let errPow = 0;
  let refPow = 0;
  for (let i = 0; i < n; i += 1) {
    const e = ref[i] - est[i];
    errPow += e * e;
    refPow += ref[i] * ref[i];
  }
  const mse = errPow / n;
  const snr = 10 * Math.log10((refPow + 1e-12) / (errPow + 1e-12));

  const frame = 320;
  const hop = 160;
  let segAcc = 0;
  let segCount = 0;
  for (let i = 0; i + frame < n; i += hop) {
    let r = 0;
    let e = 0;
    for (let j = i; j < i + frame; j += 1) {
      const d = ref[j] - est[j];
      r += ref[j] * ref[j];
      e += d * d;
    }
    segAcc += 10 * Math.log10((r + 1e-12) / (e + 1e-12));
    segCount += 1;
  }

  return { mse, snr, segSnr: segCount > 0 ? segAcc / segCount : snr };
}
