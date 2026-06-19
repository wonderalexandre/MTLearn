"""Node-attribute valuation projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import ValuationProjection


@dataclass(frozen=True)
class NodeAttributeValuation(ValuationProjection):
    """Filter and reconstruct increments of one scalar node attribute."""

    attribute: Any
    kind: str = "node_attribute"

    def key(self) -> str:
        return f"node_attribute:{getattr(self.attribute, 'name', str(self.attribute))}"

    def required_attributes(self) -> tuple[Any, ...]:
        return (self.attribute,)

    def compute_node_signal(self, tree, tree_info, *, morphology_module, attribute_dtype, device):
        attr_np = morphology_module.compute_attributes(tree, [self.attribute], dtype=attribute_dtype)[1]
        values = torch.as_tensor(attr_np, device=device).squeeze(1)
        parent = tree_info["parent"]
        parent_values = values[parent.clamp_min(0)]
        increments = values - parent_values
        root_or_self = parent == torch.arange(parent.numel(), device=parent.device)
        increments = torch.where(root_or_self, values, increments)
        alive = tree_info["tpost"] > tree_info["tpre"]
        return torch.where(alive, increments, torch.zeros_like(increments))
