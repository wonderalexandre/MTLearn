"""Normalized CFP filter specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..scoring import ScoringModel


@dataclass(frozen=True)
class NormalizedFilterSpec:
    """Validated internal representation of one CFP output specification."""

    index: int
    key: str
    tree_type: str
    tree_key: str
    attributes: tuple[Any, ...]
    scoring_model: ScoringModel
    score_sharpness: float
    preserve_root: bool
    monotonicity_weight: float
    constraint_configs: tuple[dict[str, Any], ...]
    regularizer_configs: tuple[dict[str, Any], ...]
    tos_interpolation: Any
    tos_infinity_seed_row: int
    tos_infinity_seed_col: int
