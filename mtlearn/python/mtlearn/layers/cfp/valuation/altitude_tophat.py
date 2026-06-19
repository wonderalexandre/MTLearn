"""Altitude top-hat valuation projection."""

from __future__ import annotations

import torch

from .... import morphology
from .base import ValuationProjection


class AltitudeTopHatValuation(ValuationProjection):
    """Project filtered altitude into tree-type-specific top-hat output."""

    kind = "altitude_tophat"

    def compute_node_signal(self, tree, tree_info, *, morphology_module, attribute_dtype, device):
        return tree_info["residues"]

    def requires_unfiltered_image(self) -> bool:
        return True

    def project(self, filtered_image: torch.Tensor, unfiltered_image: torch.Tensor, tree_info) -> torch.Tensor:
        tree_type = tree_info.get("tree_type")
        if tree_type == morphology.TreeType.MAX_TREE.value or tree_type == "max-tree":
            return unfiltered_image - filtered_image
        if tree_type == morphology.TreeType.MIN_TREE.value or tree_type == "min-tree":
            return filtered_image - unfiltered_image
        return torch.abs(filtered_image - unfiltered_image)
