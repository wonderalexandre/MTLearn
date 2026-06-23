"""Attribute-order score monotonicity regularization."""

from __future__ import annotations

import math
import numbers

import torch

from ._tree_info import require_complete_tree_info
from .base import Regularizer


def _validate_nonnegative_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a non-negative finite scalar.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite scalar.")
    return value


class AttributeOrderScoreMonotonicityRegularizer(Regularizer):
    """Penalize score inversions after sorting nodes by one attribute."""

    def __init__(
        self,
        weight: float = 1.0,
        *,
        feature_index: int = 0,
        direction: str = "increasing",
        min_gap: float = 0.0,
    ):
        super().__init__()
        self.weight = _validate_nonnegative_finite(weight, "weight")
        if isinstance(feature_index, bool) or not isinstance(feature_index, numbers.Integral):
            raise TypeError("feature_index must be a non-negative integer.")
        self.feature_index = int(feature_index)
        if self.feature_index < 0:
            raise ValueError("feature_index must be a non-negative integer.")
        if direction not in {"increasing", "decreasing"}:
            raise ValueError("direction must be 'increasing' or 'decreasing'.")
        self.direction = str(direction)
        self.min_gap = _validate_nonnegative_finite(min_gap, "min_gap")

    def forward(self, scores: torch.Tensor, tree_info, features=None, context=None) -> torch.Tensor:
        if features is None:
            raise ValueError("AttributeOrderScoreMonotonicityRegularizer requires normalized features.")
        if scores.dim() != 1:
            raise ValueError(f"expected scores with shape (num_nodes,), got {tuple(scores.shape)}")
        if features.dim() != 2:
            raise ValueError(f"expected features with shape (num_nodes, K), got {tuple(features.shape)}")
        if features.size(0) != scores.numel():
            raise ValueError("features and scores must have the same number of nodes.")
        if self.feature_index >= features.size(1):
            raise ValueError(f"feature_index={self.feature_index} is out of range for {features.size(1)} features.")

        _, active = require_complete_tree_info(scores, tree_info)
        values = features[:, self.feature_index].to(device=scores.device, dtype=scores.dtype)

        if int(active.sum().item()) < 2:
            return scores.sum() * 0.0

        values = values[active]
        active_scores = scores[active]
        order = torch.argsort(values)
        sorted_values = values[order]
        sorted_scores = active_scores[order]

        comparable = (sorted_values[1:] - sorted_values[:-1]) > self.min_gap
        if not bool(comparable.any().item()):
            return scores.sum() * 0.0

        if self.direction == "increasing":
            violations = torch.relu(sorted_scores[:-1] - sorted_scores[1:])
        else:
            violations = torch.relu(sorted_scores[1:] - sorted_scores[:-1])
        violations = violations[comparable]
        return self.weight * violations.square().mean()
