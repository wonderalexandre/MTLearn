"""Base class for CFP node scoring models."""

from __future__ import annotations

from typing import Any

import torch


class ScoringModel(torch.nn.Module):
    """Map node features and tree metadata to one differentiable score per node."""

    def required_features(self) -> tuple[Any, ...]:
        """Return attributes required by this scorer when known statically."""
        return ()

    def forward(self, features: torch.Tensor, tree_info=None, context=None, **kwargs) -> torch.Tensor:
        """Return node scores with shape ``(num_nodes,)``."""
        raise NotImplementedError
