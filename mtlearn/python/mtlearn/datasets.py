"""Public PyTorch dataset helpers for mtlearn."""

from __future__ import annotations

from ._datasets import GeneratedTargetImageDataset, PairedImageDataset, _split_indices
from ._datasets._split_ids import (
    SplitValidationReport,
    split_ids_by_group,
    subsets_from_ids,
    validate_split_ids,
)
from ._datasets._pair_audit import PairAudit, audit_image_pairs
from ._datasets._split_manifest import SplitManifest

GeneratedTargetImageDataset.__module__ = __name__
PairedImageDataset.__module__ = __name__
_split_indices.__module__ = __name__

__all__ = [
    "GeneratedTargetImageDataset",
    "PairedImageDataset",
    "_split_indices",
    "SplitValidationReport",
    "validate_split_ids",
    "subsets_from_ids",
    "split_ids_by_group",
    "PairAudit",
    "audit_image_pairs",
    "SplitManifest",
]


for _public in (SplitValidationReport, validate_split_ids, subsets_from_ids, split_ids_by_group,
                PairAudit, audit_image_pairs, SplitManifest):
    _public.__module__ = __name__
del _public


def __dir__() -> list[str]:
    """Return public dataset names for interactive inspection."""

    return sorted(set(globals()) | set(__all__))
