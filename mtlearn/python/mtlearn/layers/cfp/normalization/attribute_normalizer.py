"""Dataset-statistics-backed attribute normalization."""

from __future__ import annotations

import math

import torch

from . import attribute_statistics as statistics

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
        self._ds_stats = {}
        self._stats_epoch = 0
        self._stats_frozen = False
        self._moments = {}
        self._constants = {}
        self._constant_policy = None

    @property
    def ds_stats(self):
        return self._ds_stats

    @ds_stats.setter
    def ds_stats(self, value):
        self._ds_stats = value
        self.invalidate_constants()

    @property
    def stats_epoch(self):
        return self._stats_epoch

    @stats_epoch.setter
    def stats_epoch(self, value):
        self._stats_epoch = int(value)
        self.invalidate_constants()

    @property
    def stats_frozen(self):
        return self._stats_frozen

    @stats_frozen.setter
    def stats_frozen(self, value):
        self._stats_frozen = bool(value)
        self.invalidate_constants()

    def invalidate_constants(self):
        """Drop only derived scalars after an explicit statistics change."""
        self._moments.clear()
        self._constants.clear()
        self._constant_policy = None

    def summarize(self, stat_key, raw_attribute):
        """Return one independent CPU scalar summary without changing this fit."""
        summary = {}
        _update_attribute_stats(summary, self.scale_mode, stat_key, raw_attribute)
        return summary

    def merge(self, summary) -> bool:
        """Merge CPU summaries in caller order, preserving node-weighted moments.

        Repeated logical samples count repeatedly. Deduplication of retried
        tasks belongs to the later manifest coordinator, not this reducer.
        """
        if self.stats_frozen:
            return False
        summary = validate_summary(summary, self.scale_mode)
        any_changed = False
        for key, values in summary.items():
            if key not in self.ds_stats:
                self.ds_stats[key] = values
                changed = True
            elif self.scale_mode == "dataset_minmax01":
                old = self.ds_stats[key]
                changed = bool(values["amin"] < old["amin"] or values["amax"] > old["amax"])
                self.ds_stats[key] = {"amin": torch.minimum(old["amin"], values["amin"]),
                                      "amax": torch.maximum(old["amax"], values["amax"])}
            else:
                old = self.ds_stats[key]
                self.ds_stats[key] = {name: old[name] + value for name, value in values.items()}
                changed = True
            if changed:
                self.stats_epoch += 1
                any_changed = True
        return any_changed

    def prepare_constants(self, *, device="cpu", dtype=torch.float32):
        """Prepare small scalars once for the frozen snapshot and consumer type."""
        if not self.stats_frozen:
            raise RuntimeError("Freeze statistics before preparing normalization constants.")
        if self.scale_mode != "none":
            for key in self.ds_stats:
                self._constants_for(key, torch.device(device), dtype)

    def _constants_for(self, stat_key, device, dtype):
        policy = (self.scale_mode, self.eps, self.clipped_zscore_radius, self.clipped_zscore_floor)
        if policy != self._constant_policy:
            self.invalidate_constants()
            self._constant_policy = policy
        key = (stat_key, str(device), dtype)
        if key not in self._constants:
            if self.scale_mode == "dataset_minmax01":
                values = statistics._require_attribute_stats(self.ds_stats, self.scale_mode, stat_key)
                amin = values["amin"].to(device=device, dtype=dtype)
                amax = values["amax"].to(device=device, dtype=dtype)
                constants = (amin, torch.clamp(amax - amin, min=self.eps))
            else:
                moment_key = (stat_key, self.eps)
                if moment_key not in self._moments:
                    self._moments[moment_key] = statistics._zscore_moments(
                        self.ds_stats, self.scale_mode, self.eps, stat_key)
                mean, std = self._moments[moment_key]
                # Preserve the original CPU cast before transferring to MPS/CUDA.
                constants = (mean.to(dtype=dtype).to(device), std.to(dtype=dtype).to(device))
                if self.scale_mode == DATASET_CLIPPED_ZSCORE01:
                    constants += (torch.tensor(self.clipped_zscore_radius, dtype=dtype, device=device),
                                  torch.tensor(self.clipped_zscore_floor, dtype=dtype, device=device))
            self._constants[key] = constants
        return self._constants[key]

    def update(self, stat_key, raw_attribute) -> bool:
        """Update statistics for one raw node-attribute vector."""
        if self.stats_frozen:
            return False
        changed = _update_attribute_stats(self.ds_stats, self.scale_mode, stat_key, raw_attribute)
        if changed:
            self.stats_epoch += 1
        return changed

    def normalize(self, stat_key, raw_attribute):
        """Normalize with frozen scalars or the compatible mutable-fit path."""
        if self.stats_frozen and self.scale_mode != "none":
            constants = self._constants_for(stat_key, raw_attribute.device, raw_attribute.dtype)
            result = (raw_attribute - constants[0]) / constants[1]
            if self.scale_mode == DATASET_CLIPPED_ZSCORE01:
                k, floor = constants[2:]
                result = torch.clamp(result, -k, k)
                result = floor + (1.0 - floor) * ((result + k) / (2.0 * k))
            return result
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


def validate_summary(summary, scale_mode):
    """Copy and validate scalar summaries before installing or merging them."""
    from collections.abc import Mapping
    scale_mode = _validate_scale_mode(scale_mode)
    if not isinstance(summary, Mapping):
        raise TypeError("Statistics must be a mapping.")
    expected = set() if scale_mode == "none" else ({"amin", "amax"} if scale_mode == "dataset_minmax01"
                                                    else {"count", "sum", "sumsq"})
    if not expected and summary:
        raise ValueError("scale_mode='none' requires an empty statistics mapping.")
    copied = {}
    for key, values in summary.items():
        if not isinstance(key, str) or not isinstance(values, Mapping) or set(values) != expected:
            raise ValueError("Statistic keys/fields do not match the normalization mode.")
        scalars = {}
        for name, value in values.items():
            if not torch.is_tensor(value) or value.ndim != 0 or value.requires_grad:
                raise ValueError("Statistics must be detached scalar tensors.")
            value = value.detach().cpu().clone()
            if not bool(torch.isfinite(value)):
                raise ValueError("Statistics must be finite.")
            if name == "count":
                if value.dtype != torch.int64 or int(value) <= 0:
                    raise ValueError("Statistics count must be a positive int64 scalar.")
            elif value.dtype not in (torch.float32, torch.float64):
                raise ValueError("Statistics require floating moments or bounds.")
            if name in ("sum", "sumsq") and value.dtype != torch.float64:
                raise ValueError("Statistical moments must retain float64 precision.")
            scalars[name] = value
        if "amin" in scalars and (scalars["amin"].dtype != scalars["amax"].dtype or scalars["amin"] > scalars["amax"]):
            raise ValueError("Invalid min/max statistics.")
        if "sumsq" in scalars and scalars["sumsq"] < 0:
            raise ValueError("Squared moments must be nonnegative.")
        copied[key] = scalars
    return copied
