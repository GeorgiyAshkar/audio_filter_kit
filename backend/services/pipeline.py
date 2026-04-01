from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .filter_base import FilterBase


def run_pipeline(y: np.ndarray, sr: int, pipeline: Iterable[FilterBase]) -> np.ndarray:
    out = np.asarray(y, dtype=np.float32)
    for filt in pipeline:
        out = np.asarray(filt.process(out, sr), dtype=np.float32)
    return out


def total_latency_ms(sr: int, pipeline: Iterable[FilterBase]) -> float:
    samples = sum(max(0, f.latency_samples(sr)) for f in pipeline)
    return 1000.0 * samples / max(sr, 1)
