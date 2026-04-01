"""
Core backend logic for the DAS speech denoising application.

This module collects common utility functions for audio processing and
metrics, defines the base class for filters, and implements dynamic
filter discovery and pipeline execution.  By separating this logic
into its own module under ``backend``, both GUI and web frontends can
reuse the same functionality without duplication, facilitating a
microservice‑friendly architecture.

The design assumes that individual filters are implemented as Python
classes deriving from :class:`FilterBase` and located in the
``backend.filters`` package.  New filters can be added simply by
creating modules in that package without modifying this file.  The
``get_filter_classes`` function uses runtime inspection to discover
available filters.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import librosa  # type: ignore
import numpy as np
from scipy.signal import butter, lfilter, medfilt, wiener, iirnotch, welch  # type: ignore

__all__ = [
    "ensure_mono",
    "align_signals",
    "mse",
    "snr_db",
    "segmental_snr",
    "log_spectral_distance",
    "resample_if_needed",
    "rms_frames",
    "speech_mask_from_energy",
    "normalize_pairing_key",
    "FilterBase",
    "get_filter_classes",
    "run_pipeline",
    "total_latency_ms",
    "compute_metrics",
    "composite_score",
]


###############################################################################
# Audio utilities
###############################################################################


def ensure_mono(x: np.ndarray) -> np.ndarray:
    """Return a mono version of a signal by averaging channels.

    Parameters
    ----------
    x : np.ndarray
        Input audio array; may be 1D or 2D.  If 2D, channels are assumed
        along the first axis.

    Returns
    -------
    np.ndarray
        A 1D float32 array representing the mono signal.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=0)
    return x.astype(np.float32)


def align_signals(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trim two arrays to the same length.

    Parameters
    ----------
    a, b : np.ndarray
        Input signals.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Versions of ``a`` and ``b`` with equal length equal to
        ``min(len(a), len(b))``.
    """
    n = min(len(a), len(b))
    return a[:n], b[:n]


def mse(ref: np.ndarray, est: np.ndarray) -> float:
    """Mean squared error between reference and estimate."""
    ref, est = align_signals(ref, est)
    return float(np.mean((ref - est) ** 2)) if len(ref) > 0 else float("nan")


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """Signal‑to‑noise ratio in decibels."""
    ref, est = align_signals(ref, est)
    if len(ref) == 0:
        return float("nan")
    noise = est - ref
    ps = float(np.mean(ref ** 2) + 1e-12)
    pn = float(np.mean(noise ** 2) + 1e-12)
    return 10.0 * math.log10(ps / pn)


def segmental_snr(ref: np.ndarray, est: np.ndarray, sr: int, frame_ms: float = 20.0) -> float:
    """Compute segmental SNR averaged over frames of a fixed duration."""
    ref, est = align_signals(ref, est)
    if len(ref) == 0:
        return float("nan")
    frame = max(32, int(sr * frame_ms / 1000.0))
    vals: List[float] = []
    for i in range(0, len(ref) - frame + 1, frame):
        r = ref[i:i + frame]
        e = est[i:i + frame]
        n = e - r
        ps = float(np.mean(r ** 2) + 1e-12)
        pn = float(np.mean(n ** 2) + 1e-12)
        s = 10.0 * math.log10(ps / pn)
        s = max(-10.0, min(35.0, s))
        vals.append(s)
    return float(np.mean(vals)) if vals else float("nan")


def log_spectral_distance(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Compute the log‑spectral distance between two signals."""
    ref, est = align_signals(ref, est)
    if len(ref) == 0:
        return float("nan")
    n_fft = 1024
    hop = 256
    r = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop)) + 1e-8
    e = np.abs(librosa.stft(est, n_fft=n_fft, hop_length=hop)) + 1e-8
    n_frames = min(r.shape[1], e.shape[1])
    r = r[:, :n_frames]
    e = e[:, :n_frames]
    d = np.sqrt(np.mean((20.0 * np.log10(r) - 20.0 * np.log10(e)) ** 2, axis=0))
    return float(np.mean(d))


def resample_if_needed(y: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample audio if the source sample rate differs from the target."""
    if src_sr == dst_sr:
        return y
    return librosa.resample(y, orig_sr=src_sr, target_sr=dst_sr)


def rms_frames(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Compute RMS values for overlapping frames."""
    if len(y) < frame:
        pad = np.pad(y, (0, frame - len(y)))
        return np.array([np.sqrt(np.mean(pad ** 2) + 1e-12)], dtype=np.float32)
    vals: List[float] = []
    for i in range(0, len(y) - frame + 1, hop):
        vals.append(np.sqrt(np.mean(y[i:i + frame] ** 2) + 1e-12))
    return np.asarray(vals, dtype=np.float32)


def speech_mask_from_energy(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    threshold_mult: float = 1.5,
    noise_percentile: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate a speech activity mask based on frame‑level energy.

    Returns a boolean mask over frames as well as a boolean mask over
    individual samples indicating speech regions.
    """
    frame = max(32, int(sr * frame_ms / 1000.0))
    hop = max(16, int(sr * hop_ms / 1000.0))
    erms = rms_frames(y, frame, hop)
    baseline = np.percentile(erms, noise_percentile)
    thr = max(1e-6, baseline * threshold_mult)
    frame_mask = erms >= thr
    sample_mask = np.zeros(len(y), dtype=bool)
    for idx, is_speech in enumerate(frame_mask):
        start = idx * hop
        end = min(len(y), start + frame)
        sample_mask[start:end] = is_speech
    return frame_mask, sample_mask


def normalize_pairing_key(path: str) -> str:
    """Normalise a filename to a canonical form for pairing raw and reference recordings."""
    stem = os.path.splitext(os.path.basename(path))[0].lower().strip()
    stem = re.sub(r"_+$", "", stem)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"\s+", "", stem)
    return stem


###############################################################################
# Filter base class and dynamic discovery
###############################################################################


class FilterBase:
    """
    Abstract base class for all filters.

    A filter transforms an input waveform to an output waveform.  It
    stores a dictionary of parameters and exposes metadata about the
    parameter search space for brute‑force optimisation.  Subclasses
    override :meth:`apply`, optionally :meth:`latency_samples`,
    :meth:`param_spec` and :meth:`search_space`.
    """

    name: str = "Base"

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)

    def apply(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Transform a signal.  Subclasses must override."""
        return y

    def latency_samples(self, sr: int) -> int:
        """Return the number of samples delay introduced by the filter."""
        return 0

    def param_spec(self) -> Dict[str, Tuple[type, Any, Any, Any, Any]]:
        """
        Return a parameter specification dictionary.

        Each entry maps a parameter name to a tuple of
        (type, default_value, min_value, max_value, step).  Subclasses
        override this to declare their parameters.
        """
        return {}

    def search_space(self) -> List[Dict[str, Any]]:
        """Return a list of parameter dictionaries for brute‑force search."""
        return [self.params.copy()]

    def clone(self) -> "FilterBase":
        """Return a copy of this filter with the same parameters."""
        return self.__class__(**self.params.copy())

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this filter to a dictionary."""
        return {"class": self.__class__.__name__, "params": self.params.copy()}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FilterBase":
        """Deserialise a filter from a dictionary produced by :meth:`to_dict`."""
        klass_name = payload["class"]
        params = payload.get("params", {})
        klass = _FILTER_NAME_TO_CLASS.get(klass_name)
        if klass is None:
            raise ValueError(f"Unknown filter class {klass_name}")
        return klass(**params)


# Registry of available filter classes
_FILTER_CLASSES: List[Type[FilterBase]] = []
_FILTER_NAME_TO_CLASS: Dict[str, Type[FilterBase]] = {}


def _discover_filters() -> None:
    """
    Populate the registry of filter classes by scanning the
    ``backend.filters`` package.  Each module in that package may
    define one or more subclasses of :class:`FilterBase`.  Classes
    discovered in this way are added to the global registry.
    """
    global _FILTER_CLASSES, _FILTER_NAME_TO_CLASS
    if _FILTER_CLASSES:
        return
    filters_dir = os.path.join(os.path.dirname(__file__), "filters")
    if not os.path.isdir(filters_dir):
        return
    for fname in os.listdir(filters_dir):
        if fname.startswith("_") or not fname.endswith(".py"):
            continue
        mod_name = f"backend.filters.{os.path.splitext(fname)[0]}"
        try:
            module = __import__(mod_name, fromlist=["*"])
        except Exception:
            continue
        for name, obj in module.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, FilterBase) and obj is not FilterBase:
                _FILTER_CLASSES.append(obj)
    _FILTER_NAME_TO_CLASS = {cls.__name__: cls for cls in _FILTER_CLASSES}


def get_filter_classes() -> List[Type[FilterBase]]:
    """Return a list of all discovered filter classes."""
    if not _FILTER_CLASSES:
        _discover_filters()
    return list(_FILTER_CLASSES)


###############################################################################
# Pipeline utilities
###############################################################################


def run_pipeline(y: np.ndarray, sr: int, pipeline: Optional[List[FilterBase]] = None) -> np.ndarray:
    """
    Apply a sequence of filters to a signal.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    sr : int
        Sample rate of the signal.
    pipeline : list of FilterBase, optional
        Sequence of filters to apply.  If ``None`` or empty, the
        identity transform is returned.

    Returns
    -------
    np.ndarray
        The processed signal as float32.
    """
    out = np.asarray(y, dtype=np.float32).copy()
    for filt in (pipeline or []):
        out = ensure_mono(filt.apply(out, sr))
    return out.astype(np.float32)


def total_latency_ms(sr: int, pipeline: Optional[List[FilterBase]] = None) -> float:
    """Compute the total latency of a filter pipeline in milliseconds."""
    total_samples = sum(int(f.latency_samples(sr)) for f in (pipeline or []))
    return 1000.0 * total_samples / max(sr, 1)


def compute_metrics(ref: np.ndarray, est: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute basic evaluation metrics (MSE, SNR, segmental SNR, LSD)."""
    return {
        "mse": mse(ref, est),
        "snr": snr_db(ref, est),
        "seg_snr": segmental_snr(ref, est, sr),
        "lsd": log_spectral_distance(ref, est, sr),
    }


def composite_score(metrics: Dict[str, float]) -> float:
    """
    Combine multiple metrics into a single scalar score.

    Higher values correspond to better perceptual quality.  The weights
    applied here are heuristic and can be tuned for a particular
    application.
    """
    return float(
        1.5 * metrics.get("snr", 0.0)
        + 1.0 * metrics.get("seg_snr", 0.0)
        - 0.1 * metrics.get("lsd", 0.0)
        - 1000.0 * metrics.get("mse", 0.0)
    )
