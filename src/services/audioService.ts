import type { AudioTrack } from '../types';

const audioContext = new AudioContext();

export async function decodeAudioFile(file: File): Promise<AudioTrack> {
  const arr = await file.arrayBuffer();
  const decoded = await audioContext.decodeAudioData(arr);
  const ch0 = decoded.getChannelData(0);
  const mono = new Float32Array(decoded.length);
  if (decoded.numberOfChannels === 1) {
    mono.set(ch0);
  } else {
    for (let i = 0; i < decoded.length; i += 1) {
      let sum = 0;
      for (let c = 0; c < decoded.numberOfChannels; c += 1) sum += decoded.getChannelData(c)[i] ?? 0;
      mono[i] = sum / decoded.numberOfChannels;
    }
  }
  return { name: file.name, sampleRate: decoded.sampleRate, samples: mono };
}

export function cropBySelection(track: AudioTrack, selection: { start: number; end: number } | null): Float32Array {
  if (!selection) return track.samples;
  const start = Math.max(0, Math.floor(Math.min(selection.start, selection.end) * track.sampleRate));
  const end = Math.min(track.samples.length, Math.ceil(Math.max(selection.start, selection.end) * track.sampleRate));
  if (end <= start) return track.samples;
  return track.samples.slice(start, end);
}
