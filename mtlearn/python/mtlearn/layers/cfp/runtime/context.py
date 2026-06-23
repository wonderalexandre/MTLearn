"""Runtime context passed between CFP extension points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CFPContext:
    """Per-application metadata for CFP scoring, projections, and penalties."""

    sample_key: str | None = None
    batch_index: int | None = None
    channel_index: int | None = None
    mode: str | None = None
    spec_name: str | None = None
    spec_index: int | None = None
    tree_type: Any | None = None
    tree_key: str | None = None
    attribute_types: tuple[Any, ...] = ()
    attribute_names: tuple[str, ...] = ()
    image_shape: tuple[int, int] | None = None
    normalization_mode: str | None = None
    score_sharpness: float | None = None
    is_training: bool | None = None
    raw_attributes: Mapping[Any, Any] = field(default_factory=dict)
    normalized_attributes: Mapping[Any, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
