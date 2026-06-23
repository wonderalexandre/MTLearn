"""Base class for CFP node scoring models."""

from __future__ import annotations

import math
from typing import Any

import torch


class ScoringModel(torch.nn.Module):
    """Map node features and tree metadata to one differentiable score per node."""

    def required_features(self) -> tuple[Any, ...]:
        """Return attributes required by this scorer when known statically."""
        return ()

    @staticmethod
    def identity_logit(p0: float) -> float:
        """Return a clipped logit used by identity-like score initialization."""
        p0 = max(min(float(p0), 1.0 - 1e-6), 1e-6)
        return math.log(p0 / (1.0 - p0))

    def init_identity(self, *, score_sharpness: float, p0: float = 0.995, **kwargs) -> None:
        """Initialize the scorer so node scores start close to ``p0``.

        Scorers that cannot define a meaningful identity initialization should
        keep this default implementation so callers can fail explicitly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support identity initialization.")

    def forward(self, features: torch.Tensor, tree_info=None, context=None, **kwargs) -> torch.Tensor:
        """Return node scores with shape ``(num_nodes,)``."""
        raise NotImplementedError
