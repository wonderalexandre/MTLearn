"""Dataset-statistics operations for CFP attribute normalization."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch

DEFAULT_SCALE_MODE = "dataset_clipped_zscore01"
DATASET_MINMAX01 = "dataset_minmax01"
DATASET_ZSCORE = "dataset_zscore"
DATASET_CLIPPED_ZSCORE01 = "dataset_clipped_zscore01"
NONE_SCALE_MODE = "none"

SUPPORTED_SCALE_MODES = frozenset(
    {
        DATASET_CLIPPED_ZSCORE01,
        DATASET_MINMAX01,
        DATASET_ZSCORE,
        NONE_SCALE_MODE,
    }
)
STATISTICAL_SCALE_MODES = frozenset(
    {
        DATASET_CLIPPED_ZSCORE01,
        DATASET_MINMAX01,
        DATASET_ZSCORE,
    }
)


def _validate_scale_mode(scale_mode: str) -> str:
    """Return a normalized scale-mode string or raise for unsupported modes."""
    normalized = str(scale_mode)
    if normalized not in SUPPORTED_SCALE_MODES:
        supported = ", ".join(sorted(SUPPORTED_SCALE_MODES))
        raise ValueError(f"unknown scale_mode: {normalized}. Supported modes: {supported}")
    return normalized


def _update_attribute_stats(
    ds_stats: MutableMapping[Any, dict[str, torch.Tensor]],
    scale_mode: str,
    stat_key: Any,
    raw_attribute: torch.Tensor,
) -> bool:
    """Update dataset-level statistics for one raw node-attribute vector."""
    scale_mode = _validate_scale_mode(scale_mode)
    if scale_mode == DATASET_CLIPPED_ZSCORE01:
        scale_mode = DATASET_ZSCORE

    if scale_mode == DATASET_MINMAX01:
        values = raw_attribute.detach().to(device="cpu")
        amin_new = torch.min(values)
        amax_new = torch.max(values)
        changed = False
        if stat_key not in ds_stats:
            ds_stats[stat_key] = {"amin": amin_new, "amax": amax_new}
            changed = True
        else:
            stats = ds_stats[stat_key]
            stats["amin"] = stats["amin"].to(device="cpu", dtype=amin_new.dtype)
            stats["amax"] = stats["amax"].to(device="cpu", dtype=amax_new.dtype)
            if amin_new < stats["amin"]:
                stats["amin"] = amin_new
                changed = True
            if amax_new > stats["amax"]:
                stats["amax"] = amax_new
                changed = True
        return changed

    if scale_mode == DATASET_ZSCORE:
        values = raw_attribute.detach()
        count = torch.tensor(values.numel(), dtype=torch.long, device="cpu")
        values_for_stats = values.to(device="cpu").to(dtype=torch.float64)
        total = torch.sum(values_for_stats)
        squared_total = torch.sum(values_for_stats * values_for_stats)
        if stat_key not in ds_stats:
            ds_stats[stat_key] = {
                "count": count,
                "sum": total,
                "sumsq": squared_total,
            }
        else:
            stats = ds_stats[stat_key]
            stats["count"] = stats["count"].to(device="cpu", dtype=torch.long) + count
            stats["sum"] = stats["sum"].to(device="cpu", dtype=torch.float64) + total
            stats["sumsq"] = stats["sumsq"].to(device="cpu", dtype=torch.float64) + squared_total
        return True

    return False


def _normalize_with_attribute_stats(
    ds_stats: Mapping[Any, Mapping[str, torch.Tensor]],
    scale_mode: str,
    eps: float,
    stat_key: Any,
    raw_attribute: torch.Tensor,
) -> torch.Tensor:
    """Normalize one raw node-attribute vector with CFP dataset statistics."""
    scale_mode = _validate_scale_mode(scale_mode)

    if scale_mode == DATASET_CLIPPED_ZSCORE01:
        return _normalize_dataset_clipped_zscore01(ds_stats, eps, stat_key, raw_attribute)

    if scale_mode == DATASET_MINMAX01:
        stats = _require_attribute_stats(ds_stats, scale_mode, stat_key)
        amin = stats["amin"].to(dtype=raw_attribute.dtype, device=raw_attribute.device)
        amax = stats["amax"].to(dtype=raw_attribute.dtype, device=raw_attribute.device)
        denom = torch.clamp(amax - amin, min=eps)
        return (raw_attribute - amin) / denom

    if scale_mode == DATASET_ZSCORE:
        mean64, std64 = _zscore_moments(ds_stats, scale_mode, eps, stat_key)
        mean = mean64.to(dtype=raw_attribute.dtype)
        std = std64.to(dtype=raw_attribute.dtype)
        if mean.device != raw_attribute.device:
            mean = mean.to(device=raw_attribute.device)
            std = std.to(device=raw_attribute.device)
        return (raw_attribute - mean) / std

    return raw_attribute


def _require_attribute_stats(
    ds_stats: Mapping[Any, Mapping[str, torch.Tensor]],
    scale_mode: str,
    stat_key: Any,
) -> Mapping[str, torch.Tensor]:
    """Return stored stats or fail for modes that require offline fitting."""
    stats = ds_stats.get(stat_key, None)
    if stats is None:
        raise RuntimeError(
            f"scale_mode='{scale_mode}' requires dataset statistics. "
            "Call build_dataloader_cached(...) on the training split or load_stats(...) before forward/inspection."
        )
    return stats


def _normalize_dataset_clipped_zscore01(
    ds_stats: Mapping[Any, Mapping[str, torch.Tensor]],
    eps: float,
    stat_key: Any,
    raw_attribute: torch.Tensor,
    *,
    clipped_zscore_radius: float = 3.0,
    clipped_zscore_floor: float = 0.05,
) -> torch.Tensor:
    """Normalize attributes with z-score clipping followed by positive rescaling."""
    mean64, std64 = _zscore_moments(ds_stats, DATASET_CLIPPED_ZSCORE01, eps, stat_key)
    mean = mean64.to(dtype=raw_attribute.dtype)
    std = std64.to(dtype=raw_attribute.dtype)
    if mean.device != raw_attribute.device:
        mean = mean.to(device=raw_attribute.device)
        std = std.to(device=raw_attribute.device)
    x = (raw_attribute - mean) / std
    k = torch.tensor(clipped_zscore_radius, dtype=x.dtype, device=x.device)
    x = torch.clamp(x, -k, k)
    floor = torch.tensor(clipped_zscore_floor, dtype=x.dtype, device=x.device)
    return floor + (1.0 - floor) * ((x + k) / (2.0 * k))


def _zscore_moments(
    ds_stats: Mapping[Any, Mapping[str, torch.Tensor]],
    scale_mode: str,
    eps: float,
    stat_key: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return z-score mean/std from CPU float64 statistics."""
    stats = _require_attribute_stats(ds_stats, scale_mode, stat_key)
    count = stats["count"].to(device="cpu", dtype=torch.float64)
    if bool(torch.any(count <= 0).item()):
        raise RuntimeError(
            f"scale_mode='{scale_mode}' requires non-empty dataset statistics. "
            "Call build_dataloader_cached(...) on a non-empty training split or load_stats(...)."
        )
    total = stats["sum"].to(device="cpu", dtype=torch.float64)
    squared_total = stats["sumsq"].to(device="cpu", dtype=torch.float64)
    mean = total / count
    var = squared_total / count - mean * mean
    std = torch.sqrt(torch.clamp(var, min=eps))
    return mean, std


__all__ = [
    "DATASET_CLIPPED_ZSCORE01",
    "DATASET_MINMAX01",
    "DATASET_ZSCORE",
    "DEFAULT_SCALE_MODE",
    "NONE_SCALE_MODE",
    "STATISTICAL_SCALE_MODES",
    "SUPPORTED_SCALE_MODES",
]
