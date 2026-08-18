"""Public PyTorch dataset helpers for mtlearn."""

from __future__ import annotations

from ._datasets import GeneratedTargetImageDataset, PairedImageDataset, _split_indices

GeneratedTargetImageDataset.__module__ = __name__
PairedImageDataset.__module__ = __name__
_split_indices.__module__ = __name__

__all__ = [
    "GeneratedTargetImageDataset",
    "PairedImageDataset",
    "_split_indices",
]


def __dir__() -> list[str]:
    """Return public dataset names for interactive inspection."""

    return sorted(set(globals()) | set(__all__))
