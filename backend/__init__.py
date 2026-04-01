"""Backend package for the DAS Speech Filter Explorer.

This package contains all of the core functionality for the DAS
speech filtering application, including signal processing utilities,
metric calculations and a registry of available filter classes.  The
frontend should import symbols from :mod:`backend.core` and discover
filters via :func:`backend.core.get_filter_classes` rather than
importing modules directly from :mod:`backend.filters`.

The presence of this file marks this directory as a Python package.
"""

from .core import (
    ensure_mono,
    resample_if_needed,
    normalize_pairing_key,
    get_filter_classes,
    run_pipeline,
    compute_metrics,
    total_latency_ms,
    composite_score,
    FilterBase,
)

__all__ = [
    "ensure_mono",
    "resample_if_needed",
    "normalize_pairing_key",
    "get_filter_classes",
    "run_pipeline",
    "compute_metrics",
    "total_latency_ms",
    "composite_score",
    "FilterBase",
]