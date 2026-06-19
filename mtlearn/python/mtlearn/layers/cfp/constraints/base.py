"""Base class for CFP score constraints."""

from __future__ import annotations

import torch


class ScoreConstraint(torch.nn.Module):
    """Post-process node scores while keeping the scoring model independent."""

    def forward(self, scores: torch.Tensor, tree_info, context=None) -> torch.Tensor:
        """Return constrained scores."""
        raise NotImplementedError
