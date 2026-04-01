from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import stft


def _align(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(ref), len(est))
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return ref[:n].astype(np.float32), est[:n].astype(np.float32)


def compute_metrics(ref: np.ndarray, est: np.ndarray, sr: int) -> Dict[str, float]:
    ref, est = _align(ref, est)
    if len(ref) == 0:
        return {"mse": float("inf"), "snr": -120.0, "seg_snr": -120.0, "lsd": 99.0}
    err = ref - est
    mse = float(np.mean(err ** 2))
    ref_power = float(np.mean(ref ** 2) + 1e-12)
    noise_power = float(np.mean(err ** 2) + 1e-12)
    snr = float(10.0 * np.log10(ref_power / noise_power))

    frame = max(128, int(0.02 * sr))
    hop = max(64, int(0.01 * sr))
    seg_vals = []
    for i in range(0, max(1, len(ref) - frame), hop):
        r = ref[i:i + frame]
        e = err[i:i + frame]
        if len(r) < 8:
            continue
        rp = np.mean(r ** 2) + 1e-12
        ep = np.mean(e ** 2) + 1e-12
        seg_vals.append(10.0 * np.log10(rp / ep))
    seg_snr = float(np.mean(seg_vals)) if seg_vals else snr

    _, _, r_stft = stft(ref, fs=sr, nperseg=min(512, len(ref)))
    _, _, e_stft = stft(est, fs=sr, nperseg=min(512, len(est)))
    r_mag = np.abs(r_stft) + 1e-8
    e_mag = np.abs(e_stft) + 1e-8
    lsd = float(np.mean(np.sqrt(np.mean((20 * np.log10(r_mag) - 20 * np.log10(e_mag)) ** 2, axis=0))))

    return {"mse": mse, "snr": snr, "seg_snr": seg_snr, "lsd": lsd}


def composite_score(metrics: Dict[str, float]) -> float:
    return 1.2 * metrics["snr"] + 0.8 * metrics["seg_snr"] - 2.0 * metrics["lsd"] - 500.0 * metrics["mse"]
