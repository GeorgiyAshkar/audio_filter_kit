from __future__ import annotations

import os
import re

import numpy as np
import librosa  # type: ignore


def ensure_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return np.mean(y, axis=0, dtype=np.float32)
    return y.reshape(-1).astype(np.float32)


def resample_if_needed(y: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return ensure_mono(y)
    return ensure_mono(librosa.resample(ensure_mono(y), orig_sr=src_sr, target_sr=target_sr))


def normalize_pairing_key(path_or_name: str) -> str:
    base = os.path.splitext(os.path.basename(path_or_name))[0].lower().strip()
    base = re.sub(r"\s+", "_", base)
    base = re.sub(r"\(\d+\)$", "", base)
    base = re.sub(r"[_-]+\d+$", "", base)
    return base.strip("_-")
