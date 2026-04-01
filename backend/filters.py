from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt, wiener


@dataclass
class FilterConfig:
    name: str
    params: Dict[str, Any]


FILTER_LIBRARY: Dict[str, Dict[str, Any]] = {
    "Low-pass": {"cutoff_hz": [3000.0, 200.0, 12000.0, 100.0], "order": [4, 1, 10, 1]},
    "High-pass": {"cutoff_hz": [120.0, 20.0, 2000.0, 10.0], "order": [4, 1, 10, 1]},
    "Band-pass": {
        "low_hz": [120.0, 20.0, 5000.0, 10.0],
        "high_hz": [3800.0, 200.0, 12000.0, 100.0],
        "order": [4, 1, 10, 1],
    },
    "Notch": {"freq": [50.0, 30.0, 400.0, 1.0], "q": [30.0, 1.0, 80.0, 1.0]},
    "Moving-average": {"window": [9, 1, 201, 2]},
    "Median": {"kernel_size": [7, 3, 101, 2]},
    "Wiener": {"size": [9, 3, 101, 2]},
    "Gain": {"gain": [1.0, 0.1, 3.0, 0.1]},
}


def default_pipeline_item(name: str) -> FilterConfig:
    params = {k: v[0] for k, v in FILTER_LIBRARY[name].items()}
    return FilterConfig(name=name, params=params)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-12:
        return y.astype(np.float32)
    return (y / peak).astype(np.float32)


def _butter(y: np.ndarray, sr: int, btype: str, order: int, cutoff) -> np.ndarray:
    ny = sr / 2.0
    if isinstance(cutoff, tuple):
        wn = [max(1.0, cutoff[0]) / ny, min(ny - 1.0, cutoff[1]) / ny]
    else:
        wn = max(1.0, min(float(cutoff), ny - 1.0)) / ny
    b, a = butter(int(order), wn, btype=btype)
    return filtfilt(b, a, y).astype(np.float32)


def apply_filter(y: np.ndarray, sr: int, cfg: FilterConfig) -> np.ndarray:
    p = cfg.params
    if cfg.name == "Low-pass":
        return _butter(y, sr, "low", int(p["order"]), float(p["cutoff_hz"]))
    if cfg.name == "High-pass":
        return _butter(y, sr, "high", int(p["order"]), float(p["cutoff_hz"]))
    if cfg.name == "Band-pass":
        low = float(p["low_hz"])
        high = float(p["high_hz"])
        if high <= low:
            high = low + 50.0
        return _butter(y, sr, "band", int(p["order"]), (low, high))
    if cfg.name == "Notch":
        w0 = min(float(p["freq"]), sr / 2 - 5.0)
        b, a = iirnotch(w0, float(p["q"]), sr)
        return filtfilt(b, a, y).astype(np.float32)
    if cfg.name == "Moving-average":
        w = max(1, int(p["window"]))
        kern = np.ones(w, dtype=np.float32) / float(w)
        return np.convolve(y, kern, mode="same").astype(np.float32)
    if cfg.name == "Median":
        k = max(3, int(p["kernel_size"]))
        if k % 2 == 0:
            k += 1
        return medfilt(y, kernel_size=k).astype(np.float32)
    if cfg.name == "Wiener":
        s = max(3, int(p["size"]))
        if s % 2 == 0:
            s += 1
        return wiener(y, mysize=s).astype(np.float32)
    if cfg.name == "Gain":
        return (y * float(p["gain"])).astype(np.float32)
    return y.astype(np.float32)


def run_pipeline(y: np.ndarray, sr: int, pipeline: List[FilterConfig]) -> np.ndarray:
    out = y.astype(np.float32)
    for cfg in pipeline:
        out = apply_filter(out, sr, cfg)
    return normalize_audio(out)


def serialize_filter_specs() -> Dict[str, Dict[str, Any]]:
    return FILTER_LIBRARY


def pipeline_from_payload(payload: List[Dict[str, Any]]) -> List[FilterConfig]:
    result: List[FilterConfig] = []
    for item in payload:
        result.append(FilterConfig(name=item["name"], params=item.get("params", {})))
    return result


def pipeline_to_payload(items: List[FilterConfig]) -> List[Dict[str, Any]]:
    return [asdict(i) for i in items]
