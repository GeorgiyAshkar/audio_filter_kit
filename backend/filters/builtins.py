"""
Built‑in filters for DAS speech denoising.

This module defines a collection of classic digital signal processing
filters that can be used out‑of‑the‑box by the application.  Each
filter inherits from :class:`backend.core.FilterBase` and implements
the required methods for parameter specification, search space
generation and application to a signal.

Filters defined here include:

* Low‑pass, high‑pass and band‑pass Butterworth filters.
* Notch filter to remove mains hum (50/60 Hz).
* Median and moving average smoothing filters.
* Wiener filter for adaptive noise suppression.
* Temporal difference filter for emphasis of transients.
* VAD noise gate that attenuates non‑speech regions.
* Spectral subtraction filter for frequency domain denoising.
* Adaptive LMS (NLMS) and RLS filters.

Additional advanced filters are provided in ``backend.filters.advanced``.
"""

from __future__ import annotations

import numpy as np
import librosa  # type: ignore
from scipy.signal import butter, lfilter, medfilt, wiener, iirnotch

from ..core import FilterBase, speech_mask_from_energy


__all__ = [
    "LowPassFilter",
    "HighPassFilter",
    "BandPassFilter",
    "NotchFilter",
    "MedianFilter",
    "WienerFilter",
    "MovingAverageFilter",
    "TemporalDifferenceFilter",
    "VADNoiseGateFilter",
    "SpectralSubtractionFilter",
    "NLMSFilter",
    "RLSFilter",
]


###############################################################################
# Linear filters
###############################################################################


class LowPassFilter(FilterBase):
    """Butterworth low‑pass filter."""

    name = "Low‑pass"

    def __init__(self, cutoff: float = 4000.0, order: int = 4) -> None:
        super().__init__(cutoff=cutoff, order=order)

    def param_spec(self):
        return {
            "cutoff": (float, self.params.get("cutoff", 4000.0), 300.0, 10000.0, 100.0),
            "order": (int, self.params.get("order", 4), 1, 10, 1),
        }

    def search_space(self):
        return [
            {"cutoff": c, "order": o}
            for c in [2500, 3500, 4000, 5000]
            for o in [2, 4, 6]
        ]

    def apply(self, y, sr):
        cutoff = min(float(self.params["cutoff"]), sr / 2 - 50.0)
        b, a = butter(int(self.params["order"]), cutoff / (sr * 0.5), btype="low")
        return lfilter(b, a, y)


class HighPassFilter(FilterBase):
    """Butterworth high‑pass filter."""

    name = "High‑pass"

    def __init__(self, cutoff: float = 500.0, order: int = 3) -> None:
        super().__init__(cutoff=cutoff, order=order)

    def param_spec(self):
        return {
            "cutoff": (float, self.params.get("cutoff", 500.0), 20.0, 3000.0, 20.0),
            "order": (int, self.params.get("order", 3), 1, 10, 1),
        }

    def search_space(self):
        return [
            {"cutoff": c, "order": o}
            for c in [80, 150, 250, 500, 800]
            for o in [2, 3, 4]
        ]

    def apply(self, y, sr):
        cutoff = min(float(self.params["cutoff"]), sr / 2 - 50.0)
        b, a = butter(int(self.params["order"]), cutoff / (sr * 0.5), btype="high")
        return lfilter(b, a, y)


class BandPassFilter(FilterBase):
    """Butterworth band‑pass filter."""

    name = "Band‑pass"

    def __init__(self, lowcut: float = 500.0, highcut: float = 4000.0, order: int = 4) -> None:
        super().__init__(lowcut=lowcut, highcut=highcut, order=order)

    def param_spec(self):
        return {
            "lowcut": (float, self.params.get("lowcut", 500.0), 20.0, 4000.0, 20.0),
            "highcut": (float, self.params.get("highcut", 4000.0), 800.0, 12000.0, 50.0),
            "order": (int, self.params.get("order", 4), 1, 10, 1),
        }

    def search_space(self):
        lows = [80, 150, 250, 500]
        highs = [2500, 3500, 4000, 5000]
        return [
            {"lowcut": lo, "highcut": hi, "order": o}
            for lo in lows
            for hi in highs
            for o in [2, 4]
            if lo < hi
        ]

    def apply(self, y, sr):
        lo = min(float(self.params["lowcut"]), sr / 2 - 100.0)
        hi = min(float(self.params["highcut"]), sr / 2 - 50.0)
        if lo >= hi:
            lo, hi = hi - 100.0, lo + 100.0
        b, a = butter(int(self.params["order"]), [lo / (sr * 0.5), hi / (sr * 0.5)], btype="band")
        return lfilter(b, a, y)


class NotchFilter(FilterBase):
    """Notch filter (IIR) for hum removal."""

    name = "Notch 50/60 Hz"

    def __init__(self, freq: float = 50.0, q: float = 30.0) -> None:
        super().__init__(freq=freq, q=q)

    def param_spec(self):
        return {
            "freq": (float, self.params.get("freq", 50.0), 40.0, 1000.0, 1.0),
            "q": (float, self.params.get("q", 30.0), 5.0, 100.0, 1.0),
        }

    def search_space(self):
        return [
            {"freq": f, "q": q}
            for f in [50, 60, 100, 150]
            for q in [10, 20, 30, 40]
        ]

    def apply(self, y, sr):
        w0 = min(float(self.params["freq"]) / (sr * 0.5), 0.999)
        b, a = iirnotch(w0, float(self.params["q"]))
        return lfilter(b, a, y)


###############################################################################
# Smoothing and adaptive filters
###############################################################################


class MedianFilter(FilterBase):
    """Median filter for impulsive noise suppression."""

    name = "Median"

    def __init__(self, kernel_size: int = 5) -> None:
        if kernel_size % 2 == 0:
            kernel_size += 1
        super().__init__(kernel_size=kernel_size)

    def param_spec(self):
        return {
            "kernel_size": (int, self.params.get("kernel_size", 5), 3, 31, 2)
        }

    def search_space(self):
        return [{"kernel_size": k} for k in [3, 5, 7, 9]]

    def apply(self, y, sr):
        k = int(self.params["kernel_size"])
        if k % 2 == 0:
            k += 1
        return medfilt(y, k)

    def latency_samples(self, sr):
        return int(self.params["kernel_size"]) // 2


class WienerFilter(FilterBase):
    """Adaptive Wiener filter."""

    name = "Wiener"

    def __init__(self, size: int = 21) -> None:
        if size % 2 == 0:
            size += 1
        super().__init__(size=size)

    def param_spec(self):
        return {
            "size": (int, self.params.get("size", 21), 3, 201, 2)
        }

    def search_space(self):
        return [{"size": s} for s in [9, 15, 21, 31, 41]]

    def apply(self, y, sr):
        s = int(self.params["size"])
        if s % 2 == 0:
            s += 1
        return np.asarray(wiener(y, s), dtype=np.float32)

    def latency_samples(self, sr):
        return int(self.params["size"]) // 2


class MovingAverageFilter(FilterBase):
    """Simple moving average filter."""

    name = "Moving average"

    def __init__(self, window: int = 9) -> None:
        if window < 1:
            window = 1
        super().__init__(window=window)

    def param_spec(self):
        return {
            "window": (int, self.params.get("window", 9), 1, 101, 1)
        }

    def search_space(self):
        return [{"window": w} for w in [3, 5, 9, 15, 21]]

    def apply(self, y, sr):
        w = max(1, int(self.params["window"]))
        kernel = np.ones(w, dtype=np.float32) / float(w)
        return np.convolve(y, kernel, mode="same").astype(np.float32)

    def latency_samples(self, sr):
        return int(self.params["window"]) // 2


class TemporalDifferenceFilter(FilterBase):
    """Temporal difference filter emphasising transients."""

    name = "Temporal difference"

    def __init__(self, alpha: float = 0.9) -> None:
        super().__init__(alpha=alpha)

    def param_spec(self):
        return {
            "alpha": (float, self.params.get("alpha", 0.9), 0.0, 1.2, 0.05)
        }

    def search_space(self):
        return [{"alpha": a} for a in [0.5, 0.7, 0.85, 0.95]]

    def apply(self, y, sr):
        a = float(self.params["alpha"])
        out = np.empty_like(y)
        out[0] = y[0]
        out[1:] = y[1:] - a * y[:-1]
        return out.astype(np.float32)

    def latency_samples(self, sr):
        return 1


###############################################################################
# VAD‑based and spectral filters
###############################################################################


class VADNoiseGateFilter(FilterBase):
    """Noise gate using energy‑based voice activity detection."""

    name = "VAD noise gate"

    def __init__(
        self,
        threshold_mult: float = 1.5,
        silence_gain: float = 0.05,
        frame_ms: float = 20.0,
        hop_ms: float = 10.0,
    ) -> None:
        super().__init__(
            threshold_mult=threshold_mult,
            silence_gain=silence_gain,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )

    def param_spec(self):
        return {
            "threshold_mult": (float, self.params.get("threshold_mult", 1.5), 0.5, 4.0, 0.1),
            "silence_gain": (float, self.params.get("silence_gain", 0.05), 0.0, 1.0, 0.05),
            "frame_ms": (float, self.params.get("frame_ms", 20.0), 5.0, 100.0, 5.0),
            "hop_ms": (float, self.params.get("hop_ms", 10.0), 2.0, 50.0, 2.0),
        }

    def search_space(self):
        return [
            {
                "threshold_mult": t,
                "silence_gain": g,
                "frame_ms": 20.0,
                "hop_ms": 10.0,
            }
            for t in [1.0, 1.5, 2.0]
            for g in [0.0, 0.05, 0.1, 0.2]
        ]

    def apply(self, y, sr):
        _, sample_mask = speech_mask_from_energy(
            y,
            sr,
            frame_ms=float(self.params["frame_ms"]),
            hop_ms=float(self.params["hop_ms"]),
            threshold_mult=float(self.params["threshold_mult"]),
            noise_percentile=25.0,
        )
        gain = np.where(sample_mask, 1.0, float(self.params["silence_gain"]))
        return (y * gain).astype(np.float32)

    def latency_samples(self, sr):
        return int(sr * float(self.params["frame_ms"]) / 1000.0)


class SpectralSubtractionFilter(FilterBase):
    """Noise reduction via spectral subtraction."""

    name = "Spectral subtraction"

    def __init__(
        self,
        strength: float = 1.0,
        floor_ratio: float = 0.05,
        n_fft: int = 1024,
        hop_ms: float = 10.0,
        vad_threshold_mult: float = 1.5,
        noise_percentile: float = 25.0,
    ) -> None:
        super().__init__(
            strength=strength,
            floor_ratio=floor_ratio,
            n_fft=n_fft,
            hop_ms=hop_ms,
            vad_threshold_mult=vad_threshold_mult,
            noise_percentile=noise_percentile,
        )

    def param_spec(self):
        return {
            "strength": (float, self.params.get("strength", 1.0), 0.2, 3.0, 0.1),
            "floor_ratio": (float, self.params.get("floor_ratio", 0.05), 0.0, 0.5, 0.01),
            "n_fft": (int, self.params.get("n_fft", 1024), 256, 4096, 256),
            "hop_ms": (float, self.params.get("hop_ms", 10.0), 5.0, 50.0, 1.0),
            "vad_threshold_mult": (float, self.params.get("vad_threshold_mult", 1.5), 0.5, 4.0, 0.1),
            "noise_percentile": (float, self.params.get("noise_percentile", 25.0), 5.0, 50.0, 1.0),
        }

    def search_space(self):
        return [
            {
                "strength": s,
                "floor_ratio": fr,
                "n_fft": 1024,
                "hop_ms": 10.0,
                "vad_threshold_mult": vt,
                "noise_percentile": npct,
            }
            for s in [0.8, 1.0, 1.2, 1.5]
            for fr in [0.02, 0.05, 0.1]
            for vt in [1.0, 1.5, 2.0]
            for npct in [20.0, 25.0, 35.0]
        ]

    def apply(self, y, sr):
        n_fft = int(self.params["n_fft"])
        hop = max(32, int(sr * float(self.params["hop_ms"]) / 1000.0))
        win = n_fft
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)
        mag = np.abs(S)
        phase = np.angle(S)

        # Estimate noise frames using energy‑based VAD
        frame_ms = 1000.0 * n_fft / sr
        frame_mask, _ = speech_mask_from_energy(
            y,
            sr,
            frame_ms=frame_ms,
            hop_ms=float(self.params["hop_ms"]),
            threshold_mult=float(self.params["vad_threshold_mult"]),
            noise_percentile=float(self.params["noise_percentile"]),
        )
        n_frames = mag.shape[1]
        if len(frame_mask) < n_frames:
            pad = np.zeros(n_frames - len(frame_mask), dtype=bool)
            frame_mask = np.concatenate([frame_mask, pad], axis=0)
        frame_mask = frame_mask[:n_frames]
        noise_frames = ~frame_mask

        if not np.any(noise_frames):
            # fallback: use low‑energy frames
            frame_energy = np.mean(mag ** 2, axis=0)
            thr = np.percentile(frame_energy, float(self.params["noise_percentile"]))
            noise_frames = frame_energy <= thr
            if not np.any(noise_frames):
                noise_frames = np.arange(n_frames) < max(1, n_frames // 10)

        noise_profile = np.median(mag[:, noise_frames], axis=1, keepdims=True)
        strength = float(self.params["strength"])
        floor_ratio = float(self.params["floor_ratio"])
        clean_mag = np.maximum(mag - strength * noise_profile, floor_ratio * noise_profile)
        clean_S = clean_mag * np.exp(1j * phase)
        y_hat = librosa.istft(clean_S, hop_length=hop, win_length=win, length=len(y))
        return y_hat.astype(np.float32)

    def latency_samples(self, sr):
        return int(self.params["n_fft"])


###############################################################################
# Adaptive filters
###############################################################################


class NLMSFilter(FilterBase):
    """Normalised LMS adaptive filter."""

    name = "NLMS"

    def __init__(self, length: int = 32, mu: float = 0.1) -> None:
        super().__init__(length=length, mu=mu)

    def param_spec(self):
        return {
            "length": (int, self.params.get("length", 32), 4, 128, 4),
            "mu": (float, self.params.get("mu", 0.1), 0.001, 1.0, 0.01),
        }

    def search_space(self):
        return [
            {"length": l, "mu": m}
            for l in [16, 32, 64]
            for m in [0.05, 0.1, 0.2]
        ]

    def apply(self, y, sr):
        length = int(self.params["length"])
        mu = float(self.params["mu"])
        if length <= 0:
            return y
        out = np.zeros_like(y)
        w = np.zeros(length, dtype=np.float32)
        for i in range(length, len(y)):
            x = y[i - length:i][::-1]
            d = y[i]
            y_hat = np.dot(w, x)
            e = d - y_hat
            norm = np.dot(x, x) + 1e-8
            w += (mu / norm) * e * x
            out[i] = e
        out[:length] = y[:length]
        return out.astype(np.float32)

    def latency_samples(self, sr):
        return int(self.params["length"])


class RLSFilter(FilterBase):
    """Recursive least squares adaptive filter."""

    name = "RLS"

    def __init__(self, length: int = 32, lam: float = 0.99, delta: float = 0.001) -> None:
        super().__init__(length=length, lam=lam, delta=delta)

    def param_spec(self):
        return {
            "length": (int, self.params.get("length", 32), 4, 128, 4),
            "lam": (float, self.params.get("lam", 0.99), 0.9, 1.0, 0.01),
            "delta": (float, self.params.get("delta", 0.001), 1e-4, 0.1, 0.001),
        }

    def search_space(self):
        return [
            {"length": l, "lam": lam, "delta": d}
            for l in [16, 32, 64]
            for lam in [0.95, 0.98, 0.99]
            for d in [0.0001, 0.001, 0.01]
        ]

    def apply(self, y, sr):
        length = int(self.params["length"])
        lam = float(self.params["lam"])
        delta = float(self.params["delta"])
        out = np.zeros_like(y)
        w = np.zeros(length, dtype=np.float64)
        P = np.eye(length, dtype=np.float64) / delta
        for i in range(length, len(y)):
            x = y[i - length:i][::-1].astype(np.float64)
            d = float(y[i])
            z = P.dot(x)
            alpha = 1.0 / (lam + np.dot(x, z))
            k = alpha * z
            y_hat = np.dot(w, x)
            e = d - y_hat
            w += k * e
            P = (P - np.outer(k, x).dot(P)) / lam
            out[i] = e
        out[:length] = y[:length]
        return out.astype(np.float32)

    def latency_samples(self, sr):
        return int(self.params["length"])
