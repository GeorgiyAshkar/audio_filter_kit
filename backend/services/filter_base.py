from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, List, Tuple, Type


@dataclass
class FilterBase:
    """Base class for DSP filters used in a processing pipeline."""

    name: ClassVar[str] = "Base filter"
    params: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        defaults = {k: spec[1] for k, spec in self.param_spec().items()}
        defaults.update(kwargs)
        self.params = defaults

    @classmethod
    def param_spec(cls) -> Dict[str, Tuple[type, Any, Any, Any, Any]]:
        return {}

    def search_space(self) -> List[Dict[str, Any]]:
        return [dict(self.params)]

    def process(self, y, sr: int):
        return y

    def latency_samples(self, sr: int) -> int:
        return 0

    def clone(self) -> "FilterBase":
        return self.__class__(**self.params)

    def to_dict(self) -> Dict[str, Any]:
        return {"class": self.__class__.__name__, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FilterBase":
        from .registry import class_by_name

        class_name = payload.get("class")
        params = payload.get("params", {})
        filter_cls = class_by_name(class_name)
        return filter_cls(**params)
