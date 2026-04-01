from __future__ import annotations

# Ensure filter classes are registered before API functions are used.
from backend import filters as _filters  # noqa: F401

from backend.services.audio import ensure_mono, normalize_pairing_key, resample_if_needed
from backend.services.filter_base import FilterBase
from backend.services.metrics import composite_score, compute_metrics
from backend.services.pipeline import run_pipeline, total_latency_ms
from backend.services.registry import get_filter_classes

__all__ = [
    "FilterBase",
    "ensure_mono",
    "resample_if_needed",
    "normalize_pairing_key",
    "get_filter_classes",
    "run_pipeline",
    "compute_metrics",
    "total_latency_ms",
    "composite_score",
]
