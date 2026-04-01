from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter, medfilt, wiener

from backend.services.filter_base import FilterBase
from backend.services.registry import register_filter


@register_filter
class LowPassFilter(FilterBase):
    name = "Low-pass"

    @classmethod
    def param_spec(cls):
        return {
            "cutoff": (float, 3500.0, 50.0, 20000.0, 50.0),
            "order": (int, 4, 1, 10, 1),
        }

    def process(self, y, sr: int):
        cutoff = min(float(self.params["cutoff"]), sr * 0.49)
        b, a = butter(int(self.params["order"]), cutoff / (sr * 0.5), btype="low")
        return filtfilt(b, a, y)


@register_filter
class HighPassFilter(FilterBase):
    name = "High-pass"

    @classmethod
    def param_spec(cls):
        return {"cutoff": (float, 80.0, 20.0, 3000.0, 10.0), "order": (int, 4, 1, 10, 1)}

    def process(self, y, sr: int):
        cutoff = min(float(self.params["cutoff"]), sr * 0.49)
        b, a = butter(int(self.params["order"]), cutoff / (sr * 0.5), btype="high")
        return filtfilt(b, a, y)


@register_filter
class NotchFilter(FilterBase):
    name = "Notch 50/60Hz"

    @classmethod
    def param_spec(cls):
        return {"freq": (float, 50.0, 40.0, 120.0, 10.0), "q": (float, 30.0, 5.0, 80.0, 1.0)}

    def process(self, y, sr: int):
        b, a = iirnotch(float(self.params["freq"]), float(self.params["q"]), fs=sr)
        return filtfilt(b, a, y)


@register_filter
class MovingAverageFilter(FilterBase):
    name = "Moving average"

    @classmethod
    def param_spec(cls):
        return {"window": (int, 5, 1, 200, 1)}

    def process(self, y, sr: int):
        w = max(1, int(self.params["window"]))
        kernel = np.ones(w, dtype=np.float32) / w
        return lfilter(kernel, [1.0], y)

    def latency_samples(self, sr: int) -> int:
        return max(0, int(self.params["window"]) // 2)


@register_filter
class MedianFilter(FilterBase):
    name = "Median"

    @classmethod
    def param_spec(cls):
        return {"kernel_size": (int, 5, 3, 101, 2)}

    def process(self, y, sr: int):
        k = max(3, int(self.params["kernel_size"]) | 1)
        return medfilt(y, k)


@register_filter
class WienerFilter(FilterBase):
    name = "Wiener"

    @classmethod
    def param_spec(cls):
        return {"size": (int, 15, 3, 201, 2)}

    def process(self, y, sr: int):
        return wiener(y, mysize=max(3, int(self.params["size"]) | 1))
