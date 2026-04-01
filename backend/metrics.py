from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import welch


def align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def compute_metrics(reference: np.ndarray, estimate: np.ndarray, sr: int) -> Dict[str, float]:
    r, e = align(reference, estimate)
    err = r - e
    mse = float(np.mean(err**2)) if len(err) else 0.0
    s_pow = float(np.mean(r**2)) + 1e-12
    n_pow = float(np.mean(err**2)) + 1e-12
    snr = 10.0 * np.log10(s_pow / n_pow)

    frame = max(1, int(0.02 * sr))
    hops = max(1, int(0.01 * sr))
    seg = []
    for i in range(0, len(r) - frame + 1, hops):
        rs = r[i : i + frame]
        es = e[i : i + frame]
        npow = float(np.mean((rs - es) ** 2)) + 1e-12
        spow = float(np.mean(rs**2)) + 1e-12
        seg.append(10.0 * np.log10(spow / npow))
    seg_snr = float(np.mean(seg)) if seg else snr

    f1, p1 = welch(r, fs=sr, nperseg=min(1024, len(r))) if len(r) else (np.array([0.0]), np.array([1.0]))
    f2, p2 = welch(e, fs=sr, nperseg=min(1024, len(e))) if len(e) else (np.array([0.0]), np.array([1.0]))
    m = min(len(p1), len(p2))
    lsd = float(np.sqrt(np.mean((np.log10(p1[:m] + 1e-12) - np.log10(p2[:m] + 1e-12)) ** 2)))
    return {"mse": mse, "snr": float(snr), "seg_snr": float(seg_snr), "lsd": lsd}


def composite_score(m: Dict[str, float]) -> float:
    return 0.45 * m["snr"] + 0.35 * m["seg_snr"] - 0.2 * m["lsd"] - 15.0 * m["mse"]
