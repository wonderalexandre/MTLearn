"""Serialization helpers for CFP dataset normalization statistics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class StatsSerializer:
    """Move dataset-stat tensors between device state and torch-safe payloads."""

    @staticmethod
    def serialize(ds_stats: Mapping[Any, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        """Return CPU-backed, torch-safe dataset statistics."""
        return {
            str(key): {
                name: value.detach().cpu() if torch.is_tensor(value) else value
                for name, value in stats.items()
            }
            for key, stats in ds_stats.items()
        }

    @staticmethod
    def deserialize(
        serialized: Mapping[str, Mapping[str, Any]],
        *,
        device,
    ) -> dict[str, dict[str, Any]]:
        """Move serialized statistics to ``device`` for a CFP layer."""
        device = torch.device(device)
        result = {}
        for key, stats in serialized.items():
            key = str(key)
            if key.endswith("::BOX_WIDTH"):
                key = key.removesuffix("::BOX_WIDTH") + "::BOUNDING_BOX_WIDTH"
            if key in result:
                raise ValueError(f"duplicate dataset statistics for {key}")
            result[key] = {
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in stats.items()
            }
        return result
