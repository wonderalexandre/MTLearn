"""Edge score monotonicity regularization."""

from __future__ import annotations

import math
import numbers

import torch

from ._tree_info import require_complete_tree_info
from .base import Regularizer


class EdgeScoreMonotonicityRegularizer(Regularizer):
    """Penalize child scores that exceed parent scores."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        if isinstance(weight, bool) or not isinstance(weight, numbers.Real):
            raise TypeError("weight must be a non-negative finite scalar.")
        self.weight = float(weight)
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("weight must be a non-negative finite scalar.")

    def forward(self, scores: torch.Tensor, tree_info, features=None, context=None) -> torch.Tensor:
        parent, active = require_complete_tree_info(scores, tree_info)
        node_ids = torch.arange(parent.numel(), device=scores.device)
        edge_mask = active & active[parent] & (parent != node_ids)
        if not bool(edge_mask.any().item()):
            return scores.sum() * 0.0

        violations = torch.relu(scores[edge_mask] - scores[parent[edge_mask]])
        return self.weight * violations.square().mean()
