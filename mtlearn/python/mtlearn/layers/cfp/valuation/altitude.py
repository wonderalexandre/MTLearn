"""Altitude valuation projection."""

from __future__ import annotations

from .base import ValuationProjection


class AltitudeValuation(ValuationProjection):
    """Filter and reconstruct the tree altitude signal."""

    kind = "altitude"

    def compute_node_signal(self, tree, tree_info, *, morphology_module, attribute_dtype, device):
        return tree_info["residues"]
