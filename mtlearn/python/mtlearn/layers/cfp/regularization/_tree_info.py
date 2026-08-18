"""Shared tree-info validation for CFP regularizers."""

from __future__ import annotations

import torch


def require_complete_tree_info(scores: torch.Tensor, tree_info) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(parent, active)`` tensors aligned with ``scores``."""
    if scores.dim() != 1:
        raise ValueError(f"expected scores with shape (num_nodes,), got {tuple(scores.shape)}")

    required = ("parent", "tpre", "tpost")
    missing = [name for name in required if name not in tree_info]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"tree_info must contain parent, tpre, and tpost; missing: {names}")

    parent = tree_info["parent"].to(device=scores.device)
    tpre = tree_info["tpre"].to(device=scores.device)
    tpost = tree_info["tpost"].to(device=scores.device)
    if parent.numel() != scores.numel() or tpre.numel() != scores.numel() or tpost.numel() != scores.numel():
        raise ValueError("parent, tpre, tpost, and scores must have the same number of nodes.")

    # A live node owns a non-empty traversal interval in the tree tensor export.
    active = tpost > tpre
    return parent, active
