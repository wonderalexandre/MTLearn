"""Edge score monotonicity regularization."""

from __future__ import annotations

import math
import numbers

import torch

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
        parent = tree_info["parent"]
        node_ids = torch.arange(parent.numel(), device=parent.device)
        if "tpre" in tree_info and "tpost" in tree_info:
            alive = tree_info["tpost"] > tree_info["tpre"]
            edge_mask = alive & alive[parent] & (parent != node_ids)
        else:
            edge_mask = parent != node_ids
        if not bool(edge_mask.any().item()):
            return scores.sum() * 0.0

        edge_mask = edge_mask.to(device=scores.device)
        parent_on_scores = parent.to(device=scores.device)
        violations = torch.relu(scores[edge_mask] - scores[parent_on_scores[edge_mask]])
        return self.weight * violations.square().mean()
