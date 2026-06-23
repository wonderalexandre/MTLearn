"""Dataset-statistics-backed attribute normalization."""

from __future__ import annotations

import math

from .attribute_statistics import (
    DATASET_CLIPPED_ZSCORE01,
    DEFAULT_SCALE_MODE,
    _normalize_dataset_clipped_zscore01,
    _normalize_with_attribute_stats,
    _update_attribute_stats,
    _validate_scale_mode,
)


class AttributeNormalizer:
    """Thin state holder for CFP attribute normalization statistics."""

    def __init__(
        self,
        scale_mode: str = DEFAULT_SCALE_MODE,
        eps: float = 1e-6,
        *,
        clipped_zscore_radius: float = 3.0,
        clipped_zscore_floor: float = 0.05,
    ):
        self.scale_mode = _validate_scale_mode(scale_mode)
        self.eps = _positive_float(eps, "eps")
        self.clipped_zscore_radius = _positive_float(clipped_zscore_radius, "clipped_zscore_radius")
        self.clipped_zscore_floor = _bounded_float(
            clipped_zscore_floor,
            "clipped_zscore_floor",
            lower=0.0,
            upper=1.0,
        )
        self.ds_stats: dict[object, dict[str, object]] = {}
        self.stats_epoch = 0
        self.stats_frozen = False

    def update(self, stat_key, raw_attribute) -> bool:
        """Update statistics for one raw node-attribute vector."""
        if self.stats_frozen:
            return False
        changed = _update_attribute_stats(self.ds_stats, self.scale_mode, stat_key, raw_attribute)
        if changed:
            self.stats_epoch += 1
        return changed

    def normalize(self, stat_key, raw_attribute):
        """Normalize one raw node-attribute vector."""
        if self.scale_mode == DATASET_CLIPPED_ZSCORE01:
            return _normalize_dataset_clipped_zscore01(
                self.ds_stats,
                self.eps,
                stat_key,
                raw_attribute,
                clipped_zscore_radius=self.clipped_zscore_radius,
                clipped_zscore_floor=self.clipped_zscore_floor,
            )
        return _normalize_with_attribute_stats(self.ds_stats, self.scale_mode, self.eps, stat_key, raw_attribute)

    def freeze(self) -> None:
        """Stop accepting dataset-statistics updates."""
        self.stats_frozen = True

    def unfreeze(self) -> None:
        """Resume dataset-statistics updates."""
        self.stats_frozen = False

    def missing_stats(self, required_keys) -> list[str]:
        """Return required statistic keys absent from this normalizer."""
        return [key for key in required_keys if key not in self.ds_stats]


def _positive_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite positive scalar.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return normalized


def _bounded_float(value, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite scalar in [{lower}, {upper}].")
    normalized = float(value)
    if not math.isfinite(normalized) or not (lower <= normalized <= upper):
        raise ValueError(f"{name} must be a finite scalar in [{lower}, {upper}].")
    return normalized
