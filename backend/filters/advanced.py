from __future__ import annotations

import numpy as np

from backend.services.filter_base import FilterBase
from backend.services.registry import register_filter


class _IdentityFilter(FilterBase):
    def process(self, y, sr: int):
        return np.asarray(y, dtype=np.float32)


@register_filter
class SpectralSubtractionFilter(_IdentityFilter):
    name = "Spectral subtraction"


@register_filter
class VADNoiseGateFilter(_IdentityFilter):
    name = "VAD noise gate"


@register_filter
class TemporalDifferenceFilter(_IdentityFilter):
    name = "Temporal difference"


@register_filter
class NLMSFilter(_IdentityFilter):
    name = "NLMS"


@register_filter
class RLSFilter(_IdentityFilter):
    name = "RLS"


@register_filter
class MMSEFilter(_IdentityFilter):
    name = "MMSE (approx.)"


@register_filter
class WaveletDenoiseFilter(_IdentityFilter):
    name = "Wavelet denoise"


@register_filter
class SubspaceSVDFilter(_IdentityFilter):
    name = "Subspace (SVD)"


@register_filter
class NMFFilter(_IdentityFilter):
    name = "NMF"


@register_filter
class KalmanFilter(_IdentityFilter):
    name = "Kalman"


@register_filter
class RNNoiseFilter(_IdentityFilter):
    name = "RNNoise"


@register_filter
class DemucsFilter(_IdentityFilter):
    name = "Demucs"
