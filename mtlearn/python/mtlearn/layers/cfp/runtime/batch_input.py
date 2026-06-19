"""Input normalization for CFP forward-style methods."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchInput:
    """Normalized CFP batch input."""

    tensor: torch.Tensor
    index: torch.Tensor
    use_cache: bool

    def as_tuple(self):
        """Return the historical ``(x, idx, use_cache)`` representation."""
        return self.tensor, self.index, self.use_cache


class BatchInputNormalizer:
    """Normalize CFP inputs to tensor, sample index, and cache flag."""

    @staticmethod
    def normalize(value) -> BatchInput:
        """Normalize tensor/list/cached-loader input."""
        if isinstance(value, tuple) and len(value) == 2:
            return BatchInput(value[0], value[1], True)
        if (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[1], torch.Tensor)
            and value[1].dim() == 1
        ):
            return BatchInput(value[0], value[1], True)
        if isinstance(value, list):
            value = torch.stack(value, dim=0)
        return BatchInput(value, torch.arange(value.size(0), device=value.device), False)
