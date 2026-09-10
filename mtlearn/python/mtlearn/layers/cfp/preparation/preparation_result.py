"""Small result handle for finite preparation passes."""
from dataclasses import dataclass

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
