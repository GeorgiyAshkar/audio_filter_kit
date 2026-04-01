import type { FilterConfig, ProcessResponse } from '../domain/types';

export async function decodeAudio(file: File): Promise<{ data: Float32Array; sampleRate: number }> {
  const ctx = new AudioContext();
  const arr = await file.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(arr.slice(0));
  const ch = audioBuffer.getChannelData(0);
  return { data: new Float32Array(ch), sampleRate: audioBuffer.sampleRate };
}

export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

export async function processSignal(signal: Float32Array, sampleRate: number, pipeline: FilterConfig[]): Promise<ProcessResponse> {
  const response = await fetch('/api/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ signal: Array.from(signal), sampleRate, pipeline })
  });

  if (!response.ok) throw new Error('Processing failed');
  return response.json() as Promise<ProcessResponse>;
}
