"""Dataset-statistics-backed attribute normalization."""

from __future__ import annotations

import torch

from ..._helpers import normalize_with_ds_stats, update_ds_stats


class AttributeNormalizer:
    """Thin state holder for CFP attribute normalization statistics."""

    def __init__(
        self,
        scale_mode: str = "hybrid",
        eps: float = 1e-6,
        *,
        hybrid_k: float = 3.0,
        hybrid_floor_a: float = 0.05,
    ):
        self.scale_mode = str(scale_mode)
        self.eps = float(eps)
        self.hybrid_k = float(hybrid_k)
        self.hybrid_floor_a = float(hybrid_floor_a)
        self.ds_stats: dict[object, dict[str, object]] = {}
        self.stats_epoch = 0
        self.stats_frozen = False

    def update(self, stat_key, raw_attribute) -> bool:
        """Update statistics for one raw node-attribute vector."""
        if self.stats_frozen:
            return False
        smode = "zscore_tree" if self.scale_mode == "hybrid" else self.scale_mode
        changed = update_ds_stats(self.ds_stats, smode, stat_key, raw_attribute)
        if changed:
            self.stats_epoch += 1
        return changed

    def normalize(self, stat_key, raw_attribute):
        """Normalize one raw node-attribute vector."""
        if self.scale_mode != "hybrid":
            return normalize_with_ds_stats(self.ds_stats, self.scale_mode, self.eps, stat_key, raw_attribute)

        stats = self.ds_stats.get(stat_key, None)
        if stats is None:
            raise RuntimeError(
                "scale_mode='hybrid' requires dataset statistics. "
                "Call build_dataloader_cached(...) or load_stats(...) before forward/inspection."
            )
        count = stats["count"].to(dtype=stats["sum"].dtype, device=stats["sum"].device)
        mean = stats["sum"] / torch.clamp(count, min=1.0)
        var = stats["sumsq"] / torch.clamp(count, min=1.0) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=self.eps))
        x = (raw_attribute - mean) / std
        k = torch.tensor(self.hybrid_k, dtype=x.dtype, device=x.device)
        x = torch.clamp(x, -k, k)
        a = torch.tensor(self.hybrid_floor_a, dtype=x.dtype, device=x.device)
        return a + (1.0 - a) * ((x + k) / (2.0 * k))

    def freeze(self) -> None:
        """Stop accepting dataset-statistics updates."""
        self.stats_frozen = True

    def unfreeze(self) -> None:
        """Resume dataset-statistics updates."""
        self.stats_frozen = False

    def missing_stats(self, required_keys) -> list[str]:
        """Return required statistic keys absent from this normalizer."""
        return [key for key in required_keys if key not in self.ds_stats]
