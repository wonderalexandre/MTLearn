"""Small, detached snapshots of CFP normalization statistics."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Any

from .attribute_normalizer import AttributeNormalizer, validate_summary


@dataclass(frozen=True, eq=False)
class StatisticsSnapshot:
    """An immutable statistical contract with defensive copies of CPU scalars.

    ``statistics`` returns copies: callers cannot change the installed snapshot
    by mutating returned tensors. Source sample_count may be unknown for legacy
    checkpoints; it counts logical images, not tree nodes or unique contents.
    """

    contract: Mapping[str, Any]
    _statistics: Mapping[str, Any] = field(repr=False)
    sample_count: int | None = None

    def __post_init__(self):
        contract = dict(self.contract)
        expected = {"scale_mode", "eps", "clipped_zscore_radius", "clipped_zscore_floor", "attribute_dtype", "stat_keys"}
        if set(contract) != expected:
            raise ValueError("Snapshot has an invalid statistical contract.")
        if not isinstance(contract["stat_keys"], tuple) or any(not isinstance(k, str) for k in contract["stat_keys"]):
            raise ValueError("Snapshot stat_keys must be an immutable tuple of strings.")
        if contract["attribute_dtype"] not in ("float32", "float64") or len(set(contract["stat_keys"])) != len(contract["stat_keys"]):
            raise ValueError("Snapshot requires a supported attribute dtype and unique statistic keys.")
        if self.sample_count is not None and (type(self.sample_count) is not int or self.sample_count < 0):
            raise ValueError("sample_count must be a nonnegative integer or None.")
        AttributeNormalizer(contract["scale_mode"], contract["eps"],
                            clipped_zscore_radius=contract["clipped_zscore_radius"],
                            clipped_zscore_floor=contract["clipped_zscore_floor"])
        copied = validate_summary(self._statistics, contract["scale_mode"])
        required = set() if contract["scale_mode"] == "none" else set(contract["stat_keys"])
        if set(copied) != required:
            raise ValueError("Snapshot statistics do not match the required tree/attribute keys.")
        object.__setattr__(self, "contract", MappingProxyType(contract))
        object.__setattr__(self, "_statistics", MappingProxyType({k: MappingProxyType(v) for k, v in copied.items()}))

    @property
    def statistics(self):
        return {key: {name: value.clone() for name, value in values.items()}
                for key, values in self._statistics.items()}

    @classmethod
    def combine(cls, snapshots):
        """Combine snapshots in iterable order without retaining per-image data."""
        iterator = iter(snapshots)
        first = next(iterator, None)
        if not isinstance(first, cls):
            raise ValueError("combine requires at least one StatisticsSnapshot.")
        normalizer = AttributeNormalizer(first.contract["scale_mode"])
        normalizer.merge(first.statistics)
        count = first.sample_count
        for snapshot in iterator:
            if not isinstance(snapshot, cls) or snapshot.contract != first.contract:
                raise ValueError("Cannot combine incompatible statistical snapshots.")
            normalizer.merge(snapshot.statistics)
            count = None if count is None or snapshot.sample_count is None else count + snapshot.sample_count
        return cls(first.contract, normalizer.ds_stats, sample_count=count)
