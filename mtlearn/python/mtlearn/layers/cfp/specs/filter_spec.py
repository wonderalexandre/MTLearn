"""Composable CFP filter specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .feature_spec import FeatureSpec
from .tree_spec import TreeSpec


@dataclass(frozen=True)
class FilterSpec:
    """One CFP output definition assembled from extensible components."""

    name: str
    tree: TreeSpec
    features: FeatureSpec
    scoring: Any
    score_sharpness: float = 1.0
    constraints: tuple[Any, ...] = field(default_factory=tuple)
    regularizers: tuple[Any, ...] = field(default_factory=tuple)

    def all_required_attributes(self) -> tuple[Any, ...]:
        """Return scoring attributes without duplicates."""
        seen: set[Any] = set()
        required: list[Any] = []
        for attr in self.features.attributes:
            if attr not in seen:
                seen.add(attr)
                required.append(attr)
        return tuple(required)
