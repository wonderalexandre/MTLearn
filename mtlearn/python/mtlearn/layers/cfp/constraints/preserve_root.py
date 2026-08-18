"""Constraint that preserves root-node signal by forcing root score to one."""

from __future__ import annotations

import torch

from .base import ScoreConstraint


class PreserveRootConstraint(ScoreConstraint):
    """Force alive root nodes to score one."""

    def forward(self, scores: torch.Tensor, tree_info, context=None) -> torch.Tensor:
        parent = tree_info["parent"]
        node_ids = torch.arange(parent.numel(), device=parent.device)
        if "tpre" in tree_info and "tpost" in tree_info:
            root_mask = (parent == node_ids) & (tree_info["tpost"] > tree_info["tpre"])
        else:
            root_mask = parent == node_ids
        return torch.where(root_mask.to(device=scores.device), torch.ones_like(scores), scores)
