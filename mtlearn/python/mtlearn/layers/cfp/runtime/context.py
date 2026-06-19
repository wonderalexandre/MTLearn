"""Runtime context passed between CFP extension points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CFPContext:
    """Per-application metadata for CFP scoring, projections, and penalties."""

    sample_key: str | None = None
    batch_index: int | None = None
    channel_index: int | None = None
    spec_name: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
