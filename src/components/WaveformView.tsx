import { useEffect, useMemo, useRef, useState } from 'react';
import type { SelectionRange } from '../types';

type Props = {
  raw: Float32Array | null;
  reference: Float32Array | null;
  processed: Float32Array | null;
  sampleRate: number;
  selection: SelectionRange;
  onSelectionChange: (selection: SelectionRange) => void;
};

function drawWave(canvas: HTMLCanvasElement, data: Float32Array, color: string, selection: SelectionRange, sr: number) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(data.length / width));
  for (let x = 0; x < width; x += 1) {
    const idx = x * step;
    const v = data[idx] ?? 0;
    const y = ((1 - (v + 1) / 2) * height);
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (selection && sr > 0) {
    const duration = data.length / sr;
    const x1 = (Math.min(selection.start, selection.end) / duration) * width;
    const x2 = (Math.max(selection.start, selection.end) / duration) * width;
    ctx.fillStyle = 'rgba(255,255,0,0.25)';
    ctx.fillRect(x1, 0, x2 - x1, height);
  }
}

export function WaveformView({ raw, reference, processed, sampleRate, selection, onSelectionChange }: Props) {
  const panRef = useRef<HTMLCanvasElement>(null);
  const waveRef = useRef<HTMLCanvasElement>(null);
  const [dragStartX, setDragStartX] = useState<number | null>(null);

  const duration = useMemo(() => (raw && sampleRate > 0 ? raw.length / sampleRate : 0), [raw, sampleRate]);

  useEffect(() => {
    if (raw && panRef.current) drawWave(panRef.current, raw, '#9ca3af', selection, sampleRate);
  }, [raw, selection, sampleRate]);

  useEffect(() => {
    const canvas = waveRef.current;
    if (!canvas) return;
    const merged = raw ?? new Float32Array(1);
    drawWave(canvas, merged, '#60a5fa', selection, sampleRate);
    if (reference) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.globalAlpha = 0.5;
        drawWave(canvas, reference, '#10b981', selection, sampleRate);
        ctx.globalAlpha = 1;
      }
    }
    if (processed) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.globalAlpha = 0.6;
        drawWave(canvas, processed, '#f97316', selection, sampleRate);
        ctx.globalAlpha = 1;
      }
    }
  }, [raw, reference, processed, selection, sampleRate]);

  const updateSelectionFromCanvas = (x1: number, x2: number, canvas: HTMLCanvasElement) => {
    if (!raw || duration <= 0) return;
    const start = (Math.min(x1, x2) / canvas.width) * duration;
    const end = (Math.max(x1, x2) / canvas.width) * duration;
    if (Math.abs(end - start) < 0.01) {
      onSelectionChange(null);
      return;
    }
    onSelectionChange({ start, end });
  };

  return (
    <div className="plot-panel">
      <h3>Overview (panner)</h3>
      <canvas
        ref={panRef}
        width={900}
        height={100}
        onMouseDown={(e) => setDragStartX(e.nativeEvent.offsetX)}
        onMouseUp={(e) => {
          if (dragStartX === null || !panRef.current) return;
          updateSelectionFromCanvas(dragStartX, e.nativeEvent.offsetX, panRef.current);
          setDragStartX(null);
        }}
      />
      <h3>Waveform</h3>
      <canvas
        ref={waveRef}
        width={900}
        height={280}
        onMouseDown={(e) => setDragStartX(e.nativeEvent.offsetX)}
        onMouseUp={(e) => {
          if (dragStartX === null || !waveRef.current) return;
          updateSelectionFromCanvas(dragStartX, e.nativeEvent.offsetX, waveRef.current);
          setDragStartX(null);
        }}
      />
      <p>Selection: {selection ? `${selection.start.toFixed(2)}s - ${selection.end.toFixed(2)}s` : 'full track'}</p>
    </div>
  );
}
