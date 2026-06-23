"""Internal implementations for :mod:`mtlearn.datasets`."""

from __future__ import annotations

from ._generated_target import GeneratedTargetImageDataset
from ._paired_image import PairedImageDataset
from ._split import _split_indices

__all__ = [
    "GeneratedTargetImageDataset",
    "PairedImageDataset",
    "_split_indices",
]
