"""Base class for CFP regularizers."""

from __future__ import annotations

import torch


class Regularizer(torch.nn.Module):
    """Compute a scalar training penalty from CFP intermediate values."""

    def forward(self, scores: torch.Tensor, tree_info, features=None, context=None) -> torch.Tensor:
        """Return a scalar penalty tensor."""
        raise NotImplementedError
