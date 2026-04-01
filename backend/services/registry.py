from __future__ import annotations

from typing import Dict, List, Type

from .filter_base import FilterBase


_REGISTRY: Dict[str, Type[FilterBase]] = {}


def register_filter(cls: Type[FilterBase]) -> Type[FilterBase]:
    _REGISTRY[cls.__name__] = cls
    return cls


def get_filter_classes() -> List[Type[FilterBase]]:
    return sorted(_REGISTRY.values(), key=lambda c: c.name.lower())


def class_by_name(class_name: str) -> Type[FilterBase]:
    if class_name not in _REGISTRY:
        raise ValueError(f"Unknown filter class: {class_name}")
    return _REGISTRY[class_name]
