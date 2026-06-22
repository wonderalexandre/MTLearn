"""Provider for CFP tree payloads."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

import mtlearn
from .... import morphology
from ..._helpers import build_tree


class TreePayloadProvider:
    """Build tree tensors and normalized attributes."""

    def __init__(
        self,
        *,
        tree_spec_by_key: Mapping[str, Any],
        scoring_attrs_by_tree_key: Mapping[str, set[Any]],
        normalizer,
        stat_key_fn: Callable[[str, Any], str],
        device,
        attribute_dtype,
        morphology_module=morphology,
    ):
        self.tree_spec_by_key = tree_spec_by_key
        self.scoring_attrs_by_tree_key = scoring_attrs_by_tree_key
        self.normalizer = normalizer
        self.stat_key_fn = stat_key_fn
        self.device = torch.device(device)
        self.attribute_dtype = attribute_dtype
        self.morphology = morphology_module

    def build_tree(self, image_np, spec):
        """Build the morphology tree for one normalized filter spec."""
        return build_tree(
            image_np,
            spec.tree_type,
            tos_interpolation=spec.tos_interpolation,
            tos_infinity_seed_row=spec.tos_infinity_seed_row,
            tos_infinity_seed_col=spec.tos_infinity_seed_col,
        )

    def compute_tree_info(self, tree, spec):
        """Return dense tree tensors used by CFP reconstruction."""
        residues, tpre, tpost, parent, node_of_pixel = (
            mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
        )
        return {
            "residues": residues.to(self.device),
            "tpre": tpre.to(self.device),
            "tpost": tpost.to(self.device),
            "parent": parent.to(self.device),
            "node_of_pixel": node_of_pixel.to(self.device),
            "numRows": tree.numRows,
            "numCols": tree.numCols,
            "tree_type": spec.tree_type,
            "order_forward": torch.argsort(tpre, descending=False).to(self.device),
            "order_backward": torch.argsort(tpre).to(self.device),
        }

    def compute_payload(self, image_np, tree_key: str, *, update_stats: bool):
        """Build one tree payload for a channel image and tree key."""
        spec = self.tree_spec_by_key[tree_key]
        tree = self.build_tree(image_np, spec)
        info = self.compute_tree_info(tree, spec)

        base_attrs = {}
        norm_attrs = {}
        for attr_type in self.scoring_attrs_by_tree_key.get(tree_key, ()):
            attr_np = self.morphology.compute_attributes(tree, [attr_type], dtype=self.attribute_dtype)[1]
            raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
            stat_key = self.stat_key_fn(tree_key, attr_type)
            if update_stats:
                self.normalizer.update(stat_key, raw_1d)
            base_attrs[attr_type] = raw_1d.unsqueeze(1)
            norm_attrs[attr_type] = self.normalizer.normalize(stat_key, raw_1d)

        return {
            "info": info,
            "base_attrs": base_attrs,
            "norm_attrs": norm_attrs,
        }

    def get_payload(self, sample_key: str, image_channel, tree_spec, *, use_cache: bool):
        """Return a payload mapping with tree info and normalized attributes."""
        raise NotImplementedError
