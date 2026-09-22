"""Small result handle for finite preparation passes."""
from __future__ import annotations
from dataclasses import dataclass, field

from ..normalization.statistics_snapshot import StatisticsSnapshot


@dataclass(frozen=True)
class PreparationResult:
    """Progress and optional frozen statistics; never a list of tensor payloads."""
    store_path: str | None
    manifest: str | None
    sample_count: int
    status: str
    statistics: StatisticsSnapshot | None = None
    statistics_id: str | None = None


@dataclass(frozen=True)
class DiskPreparationResult(PreparationResult):
    mode: str = "reuse"
    total_samples: int = 0
    reused_entries: int = 0
    prepared_entries: int = 0
    repaired_entries: int = 0
    recovered_entries: int = 0
    statistics_source: str | None = None
    reason: str | None = None
    resources: dict = field(default_factory=dict)
