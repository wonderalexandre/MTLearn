"""Base class for CFP valuation projections."""

from __future__ import annotations

from typing import Any

import torch


class ValuationProjection:
    """Define which node signal is filtered and how reconstruction is projected."""

    kind = "valuation"

    def key(self) -> str:
        """Return the stable cache key for this valuation."""
        return self.kind

    def required_attributes(self) -> tuple[Any, ...]:
        """Return scalar attributes needed to compute this valuation."""
        return ()

    def compute_node_signal(self, tree, tree_info, *, morphology_module, attribute_dtype, device) -> torch.Tensor:
        """Compute the unfiltered node signal cached for this valuation."""
        raise NotImplementedError

    def node_signal(self, payload, tree_info) -> torch.Tensor:
        """Return one scalar signal per tree node before score filtering."""
        return payload["valuation_increments"][self.key()]

    def requires_unfiltered_image(self) -> bool:
        """Return whether ``project`` needs the unfiltered reconstructed image."""
        return False

    def project(self, filtered_image: torch.Tensor, unfiltered_image: torch.Tensor | None, tree_info) -> torch.Tensor:
        """Return the final image for this valuation."""
        return filtered_image
