"""
Advanced filters for DAS speech denoising.

This module defines more sophisticated algorithms beyond the basic
linear and smoothing filters.  Implementations here include a
simple approximation of the short‑time spectral amplitude (STSA)
MMSE estimator, wavelet shrinkage denoising, subspace methods via
singular value decomposition, nonnegative matrix factorisation,
Kalman filtering, and placeholders for deep‑learning‑based models
(RNNoise and Demucs).  Each filter class derives from
:class:`backend.core.FilterBase` and must implement the
``apply`` method along with optional parameter specifications.

Some of these filters depend on optional third‑party packages (e.g.
PyWavelets, scikit‑learn, RNNoise, Demucs).  If a dependency is
unavailable at runtime, the filter falls back to returning the input
signal unchanged.  This design supports deployment in environments
where not all libraries are installed, while preserving the API
consistency for the frontend.
"""

from __future__ import annotations

import math
import numpy as np
import librosa  # type: ignore
from scipy.signal import medfilt  # type: ignore

from ..core import FilterBase, speech_mask_from_energy

__all__ = [
    "MMSEFilter",
    "WaveletDenoiseFilter",
    "SubspaceFilter",
    "NMFNoiseFilter",
    "KalmanFilter",
    "RNNoiseFilter",
    "DemucsFilter",
]


class MMSEFilter(FilterBase):
    """Approximate short‑time spectral amplitude MMSE filter."""

    name = "MMSE (approx.)"

    def __init__(
        self,
        alpha: float = 0.98,
        floor_ratio: float = 0.05,
        n_fft: int = 1024,
        hop_ms: float = 10.0,
        vad_threshold_mult: float = 1.5,
        noise_percentile: float = 25.0,
    ) -> None:
        super().__init__(
            alpha=alpha,
            floor_ratio=floor_ratio,
            n_fft=n_fft,
            hop_ms=hop_ms,
            vad_threshold_mult=vad_threshold_mult,
            noise_percentile=noise_percentile,
        )

    def param_spec(self):
        return {
            "alpha": (float, self.params.get("alpha", 0.98), 0.7, 0.999, 0.02),
            "floor_ratio": (float, self.params.get("floor_ratio", 0.05), 0.0, 0.5, 0.01),
            "n_fft": (int, self.params.get("n_fft", 1024), 256, 4096, 256),
            "hop_ms": (float, self.params.get("hop_ms", 10.0), 5.0, 50.0, 1.0),
            "vad_threshold_mult": (float, self.params.get("vad_threshold_mult", 1.5), 0.5, 4.0, 0.1),
            "noise_percentile": (float, self.params.get("noise_percentile", 25.0), 5.0, 50.0, 1.0),
        }

    def search_space(self):
        return [
            {
                "alpha": a,
                "floor_ratio": fr,
                "n_fft": 1024,
                "hop_ms": 10.0,
                "vad_threshold_mult": vt,
                "noise_percentile": npct,
            }
            for a in [0.9, 0.95, 0.98]
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

        # Estimate noise PSD using a simple VAD
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
            frame_energy = np.mean(mag ** 2, axis=0)
            thr = np.percentile(frame_energy, float(self.params["noise_percentile"]))
            noise_frames = frame_energy <= thr
            if not np.any(noise_frames):
                noise_frames = np.arange(n_frames) < max(1, n_frames // 10)
        noise_psd = np.mean(mag[:, noise_frames] ** 2, axis=1, keepdims=True) + 1e-12
        alpha = float(self.params["alpha"])
        floor_ratio = float(self.params["floor_ratio"])
        prev_gain = np.ones((mag.shape[0], 1), dtype=np.float32)
        prev_gamma = np.ones((mag.shape[0], 1), dtype=np.float32)
        clean_mag = np.zeros_like(mag)
        for k in range(n_frames):
            yk = mag[:, [k]] ** 2
            gamma_k = yk / noise_psd
            xi = alpha * (prev_gain ** 2) * prev_gamma + (1.0 - alpha) * np.maximum(gamma_k - 1.0, 0.0)
            gain = xi / (1.0 + xi)
            gain = np.maximum(gain, floor_ratio)
            clean_mag[:, k:k + 1] = gain * mag[:, k:k + 1]
            prev_gain = gain
            prev_gamma = gamma_k
        clean_S = clean_mag * np.exp(1j * phase)
        y_hat = librosa.istft(clean_S, hop_length=hop, win_length=win, length=len(y))
        return y_hat.astype(np.float32)

    def latency_samples(self, sr):
        return int(self.params["n_fft"])


class WaveletDenoiseFilter(FilterBase):
    """Wavelet shrinkage denoising using PyWavelets (if available)."""

    name = "Wavelet denoise"

    def __init__(self, wavelet: str = "db8", level: int = 4, threshold: float = 0.5) -> None:
        super().__init__(wavelet=wavelet, level=level, threshold=threshold)

    def param_spec(self):
        return {
            "wavelet": (str, self.params.get("wavelet", "db8"), None, None, None),
            "level": (int, self.params.get("level", 4), 1, 8, 1),
            "threshold": (float, self.params.get("threshold", 0.5), 0.0, 2.0, 0.1),
        }

    def search_space(self):
        return [
            {"wavelet": w, "level": l, "threshold": t}
            for w in ["db4", "db8", "sym4", "sym8"]
            for l in [2, 3, 4]
            for t in [0.3, 0.5, 0.7]
        ]

    def apply(self, y, sr):
        try:
            import pywt  # type: ignore
        except ImportError:
            return np.asarray(y, dtype=np.float32)
        wavelet = self.params.get("wavelet", "db8")
        level = int(self.params.get("level", 4))
        thresh_mult = float(self.params.get("threshold", 0.5))
        coeffs = pywt.wavedec(y, wavelet, level=level)
        detail_coeffs = coeffs[-1]
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745
        thresh = thresh_mult * sigma * math.sqrt(2.0 * math.log(len(y) + 1))
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, thresh, mode="soft"))
        y_hat = pywt.waverec(new_coeffs, wavelet)
        y_hat = y_hat[: len(y)]
        return np.asarray(y_hat, dtype=np.float32)

    def latency_samples(self, sr):
        return 0


class SubspaceFilter(FilterBase):
    """Signal subspace denoising using SVD."""

    name = "Subspace (SVD)"

    def __init__(self, n_components: int = 6, n_fft: int = 1024, hop_ms: float = 10.0) -> None:
        super().__init__(n_components=n_components, n_fft=n_fft, hop_ms=hop_ms)

    def param_spec(self):
        return {
            "n_components": (int, self.params.get("n_components", 6), 1, 50, 1),
            "n_fft": (int, self.params.get("n_fft", 1024), 256, 4096, 256),
            "hop_ms": (float, self.params.get("hop_ms", 10.0), 5.0, 50.0, 1.0),
        }

    def search_space(self):
        return [
            {"n_components": k, "n_fft": 1024, "hop_ms": 10.0}
            for k in [4, 6, 8, 10]
        ]

    def apply(self, y, sr):
        n_fft = int(self.params["n_fft"])
        hop = max(32, int(sr * float(self.params["hop_ms"]) / 1000.0))
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        mag = np.abs(S)
        phase = np.angle(S)
        try:
            U, s, Vt = np.linalg.svd(mag, full_matrices=False)
        except np.linalg.LinAlgError:
            return y.astype(np.float32)
        k = max(1, int(self.params.get("n_components", 6)))
        k = min(k, len(s))
        S_recon = (U[:, :k] * s[:k]) @ Vt[:k, :]
        clean_S = S_recon * np.exp(1j * phase)
        y_hat = librosa.istft(clean_S, hop_length=hop, length=len(y))
        return np.asarray(y_hat, dtype=np.float32)

    def latency_samples(self, sr):
        return int(self.params["n_fft"])


class NMFNoiseFilter(FilterBase):
    """Nonnegative matrix factorisation denoising."""

    name = "NMF"

    def __init__(
        self,
        n_components: int = 8,
        speech_ratio: float = 0.5,
        n_fft: int = 1024,
        hop_ms: float = 10.0,
    ) -> None:
        super().__init__(
            n_components=n_components,
            speech_ratio=speech_ratio,
            n_fft=n_fft,
            hop_ms=hop_ms,
        )

    def param_spec(self):
        return {
            "n_components": (int, self.params.get("n_components", 8), 2, 64, 1),
            "speech_ratio": (float, self.params.get("speech_ratio", 0.5), 0.1, 0.9, 0.1),
            "n_fft": (int, self.params.get("n_fft", 1024), 256, 4096, 256),
            "hop_ms": (float, self.params.get("hop_ms", 10.0), 5.0, 50.0, 1.0),
        }

    def search_space(self):
        return [
            {"n_components": n, "speech_ratio": r, "n_fft": 1024, "hop_ms": 10.0}
            for n in [4, 8, 12, 16]
            for r in [0.3, 0.5, 0.7]
        ]

    def apply(self, y, sr):
        try:
            from sklearn.decomposition import NMF  # type: ignore
        except Exception:
            return y.astype(np.float32)
        n_fft = int(self.params["n_fft"])
        hop = max(32, int(sr * float(self.params["hop_ms"]) / 1000.0))
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        mag = np.abs(S) + 1e-8
        phase = np.angle(S)
        n_components = max(2, int(self.params.get("n_components", 8)))
        speech_ratio = float(self.params.get("speech_ratio", 0.5))
        model = NMF(n_components=n_components, init="random", random_state=0, max_iter=100)
        W = model.fit_transform(mag)
        H = model.components_
        energy = (W @ H).sum(axis=0)
        sorted_idx = np.argsort(-energy)
        keep_count = max(1, int(n_components * speech_ratio))
        keep_idx = sorted_idx[:keep_count]
        W_keep = W[:, keep_idx]
        H_keep = H[keep_idx, :]
        mag_recon = np.dot(W_keep, H_keep)
        mag_recon = np.maximum(mag_recon, 0.0)
        clean_S = mag_recon * np.exp(1j * phase)
        y_hat = librosa.istft(clean_S, hop_length=hop, length=len(y))
        return y_hat.astype(np.float32)

    def latency_samples(self, sr):
        return int(self.params["n_fft"])


class KalmanFilter(FilterBase):
    """Simple Kalman filter for smoothing a noisy signal."""

    name = "Kalman"

    def __init__(self, process_noise: float = 1e-5, measurement_noise: float = 1e-2) -> None:
        super().__init__(process_noise=process_noise, measurement_noise=measurement_noise)

    def param_spec(self):
        return {
            "process_noise": (float, self.params.get("process_noise", 1e-5), 1e-6, 1e-2, 1e-5),
            "measurement_noise": (float, self.params.get("measurement_noise", 1e-2), 1e-4, 1e-1, 1e-3),
        }

    def search_space(self):
        return [
            {"process_noise": Q, "measurement_noise": R}
            for Q in [1e-6, 1e-5, 1e-4]
            for R in [1e-3, 1e-2, 1e-1]
        ]

    def apply(self, y, sr):
        Q = float(self.params.get("process_noise", 1e-5))
        R = float(self.params.get("measurement_noise", 1e-2))
        x_est = 0.0
        P = 1.0
        out = np.zeros_like(y, dtype=np.float32)
        for i, z in enumerate(y):
            x_pred = x_est
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x_est = x_pred + K * (z - x_pred)
            P = (1.0 - K) * P_pred
            out[i] = x_est
        return out

    def latency_samples(self, sr):
        return 0


class RNNoiseFilter(FilterBase):
    """Stub for the RNNoise deep denoising model."""

    name = "RNNoise"

    def __init__(self, strength: float = 1.0) -> None:
        super().__init__(strength=strength)

    def param_spec(self):
        return {
            "strength": (float, self.params.get("strength", 1.0), 0.5, 2.0, 0.1)
        }

    def search_space(self):
        return [{"strength": s} for s in [0.8, 1.0, 1.2]]

    def apply(self, y, sr):
        strength = float(self.params.get("strength", 1.0))
        func = None
        # Attempt to use rnnoise package if available
        try:
            import rnnoise  # type: ignore
            if hasattr(rnnoise, "filter"):
                func = rnnoise.filter
        except Exception:
            pass
        if func is None:
            # Try alternate bindings
            try:
                import pyrnnoise  # type: ignore
                if hasattr(pyrnnoise, "RNNoise"):
                    model = pyrnnoise.RNNoise()
                    def func(x, sr):
                        return model.process_buffer(x)
            except Exception:
                pass
        if func is not None:
            try:
                norm = np.clip(y / max(np.max(np.abs(y)), 1e-8), -1.0, 1.0)
                pcm = (norm * 32767.0).astype(np.int16)
                filtered = func(pcm, sr)
                filtered = np.asarray(filtered, dtype=np.float32) / 32768.0
                return y + strength * (filtered - y)
            except Exception:
                pass
        # Fallback: return input unchanged
        return y.astype(np.float32)

    def latency_samples(self, sr):
        return 0


class DemucsFilter(FilterBase):
    """Stub for Demucs/Demucs‑Denoiser model."""

    name = "Demucs"

    def __init__(self, strength: float = 1.0) -> None:
        super().__init__(strength=strength)

    def param_spec(self):
        return {
            "strength": (float, self.params.get("strength", 1.0), 0.5, 2.0, 0.1)
        }

    def search_space(self):
        return [{"strength": s} for s in [0.8, 1.0, 1.2]]

    def apply(self, y, sr):
        strength = float(self.params.get("strength", 1.0))
        try:
            import torch  # type: ignore
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except Exception:
            return y.astype(np.float32)
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(name="htdemucs").to(device)
            wav = torch.from_numpy(y).float().unsqueeze(0).to(device)
            with torch.no_grad():
                enhanced = apply_model(model, wav, split=True, overlap=0.5)[0]
            enhanced = enhanced.cpu().numpy().reshape(-1)
            enhanced = enhanced[: len(y)]
            return y + strength * (enhanced - y)
        except Exception:
            return y.astype(np.float32)

    def latency_samples(self, sr):
        return 0
