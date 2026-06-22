"""Public PyTorch dataset helpers for mtlearn.

This module is the stable import surface for dataset utilities. It re-exports
the actively used image datasets and keeps the legacy attribute-filter dataset
available on demand. Public classes deliberately report ``__module__`` as
``mtlearn.datasets`` so documentation, reprs, and downstream imports point to
the stable facade rather than the internal implementation package.
"""

from __future__ import annotations

from ._datasets import GeneratedTargetImageDataset, PairedImageDataset, _split_indices

GeneratedTargetImageDataset.__module__ = __name__
PairedImageDataset.__module__ = __name__
_split_indices.__module__ = __name__

__all__ = [
    "AttributeFilterDataset",
    "GeneratedTargetImageDataset",
    "PairedImageDataset",
    "_split_indices",
]


def __getattr__(name: str):
    """Load legacy dataset helpers only when requested."""

    if name == "AttributeFilterDataset":
        from ._datasets import AttributeFilterDataset

        AttributeFilterDataset.__module__ = __name__
        globals()[name] = AttributeFilterDataset
        return AttributeFilterDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public dataset names for interactive inspection."""

    return sorted(set(globals()) | set(__all__))
