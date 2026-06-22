"""Internal implementations for :mod:`mtlearn.datasets`.

The public ``mtlearn.datasets`` module is a compatibility facade. The concrete
dataset implementations live here so the public module can stay small while
still exposing stable class names. Legacy helpers are imported lazily to avoid
loading morphology-specific dependencies unless users ask for them.
"""

from __future__ import annotations

from ._generated_target import GeneratedTargetImageDataset
from ._paired_image import PairedImageDataset
from ._split import _split_indices

__all__ = [
    "AttributeFilterDataset",
    "GeneratedTargetImageDataset",
    "PairedImageDataset",
    "_split_indices",
]


def __getattr__(name: str):
    """Lazily expose legacy dataset classes from the internal package."""

    if name == "AttributeFilterDataset":
        from ._attribute_filter import AttributeFilterDataset

        globals()[name] = AttributeFilterDataset
        return AttributeFilterDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
