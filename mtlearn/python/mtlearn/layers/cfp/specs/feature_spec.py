"""Node-feature specification for CFP scoring models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    """Attributes and normalization policy consumed by a scoring model."""

    attributes: tuple[Any, ...]
    normalization: str | None = None

    def __post_init__(self) -> None:
        if len(self.attributes) < 1:
            raise ValueError("FeatureSpec requires at least one attribute.")
