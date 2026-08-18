"""Path score monotonicity regularization."""

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


class PathScoreMonotonicityRegularizer(Regularizer):
    """Penalize descendants that score higher than their ancestors."""

    def __init__(self, weight: float = 1.0, *, max_depth: int | None = None):
        super().__init__()
        self.weight = _validate_nonnegative_finite(weight, "weight")
        if max_depth is not None:
            if isinstance(max_depth, bool) or not isinstance(max_depth, numbers.Integral):
                raise TypeError("max_depth must be a positive integer or None.")
            max_depth = int(max_depth)
            if max_depth < 1:
                raise ValueError("max_depth must be a positive integer or None.")
        self.max_depth = max_depth

    def forward(self, scores: torch.Tensor, tree_info, features=None, context=None) -> torch.Tensor:
        parent, active = require_complete_tree_info(scores, tree_info)
        node_ids = torch.arange(parent.numel(), device=scores.device)
        # ``parent.numel()`` is a safe upper bound for "all ancestors".
        max_steps = parent.numel() if self.max_depth is None else self.max_depth
        current_ancestor = parent.clone()
        active_descendant = active.clone()
        violations = []

        for _ in range(int(max_steps)):
            valid = active_descendant & active[current_ancestor] & (current_ancestor != node_ids)
            if not bool(valid.any().item()):
                break
            violations.append(torch.relu(scores[valid] - scores[current_ancestor[valid]]))

            next_ancestor = parent[current_ancestor]
            active_descendant = active_descendant & (next_ancestor != current_ancestor)
            current_ancestor = next_ancestor

        if not violations:
            return scores.sum() * 0.0

        all_violations = torch.cat(violations)
        return self.weight * all_violations.square().mean()
