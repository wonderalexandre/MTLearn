"""Morphology-tree construction specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TreeSpec:
    """Tree type and construction options shared by one or more CFP filters."""

    tree_type: Any
    tos_interpolation: Any = None
    tos_infinity_seed_row: int = 0
    tos_infinity_seed_col: int = 0

    def cache_key(self) -> str:
        """Return the stable cache key used for tree payload reuse."""
        interpolation = getattr(self.tos_interpolation, "name", str(self.tos_interpolation))
        return (
            f"{self.tree_type}|{interpolation}|"
            f"{int(self.tos_infinity_seed_row)}|{int(self.tos_infinity_seed_col)}"
        )
