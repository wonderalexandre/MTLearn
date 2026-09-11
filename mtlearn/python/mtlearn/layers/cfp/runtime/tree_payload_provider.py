"""Compatibility adapter from raw CPU preparation to legacy CFP payloads."""
from __future__ import annotations

import torch
from .... import morphology
from ..preparation import CFPPreprocessor
from ..specs import FeatureSpec, TreeSpec


class TreePayloadProvider(CFPPreprocessor):
    """Adapt model-independent morphology to the existing device cache contract."""

    def __init__(self, *, tree_spec_by_key, scoring_attrs_by_tree_key, normalizer,
                 stat_key_fn, device, attribute_dtype, morphology_module=morphology):
        super().__init__(
            tree_specs={key: TreeSpec(spec.tree_type, spec.tos_interpolation,
                                     spec.tos_infinity_seed_row, spec.tos_infinity_seed_col)
                        for key, spec in tree_spec_by_key.items()},
            features={key: FeatureSpec(tuple(sorted(attrs, key=lambda a: a.name)))
                      for key, attrs in scoring_attrs_by_tree_key.items()},
            attribute_dtype=attribute_dtype, morphology_module=morphology_module,
        )
        self.normalizer = normalizer
        self.stat_key_fn = stat_key_fn
        self.device = torch.device(device)

    def compute_payload(self, image_np, tree_key: str, *, update_stats: bool):
        prepared = self.prepare_u8(image_np, tree_key)
        return self.consume_prepared(prepared, update_stats=update_stats)

    def consume_prepared(self, prepared, *, update_stats=False):
        """Normalize and transfer only required data; never retain the result."""
        tree_key = prepared.tree_key
        info = {key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in prepared.info.items()}
        base_attrs, norm_attrs = {}, {}
        for attr_type in self.features[tree_key].attributes:
            values = prepared.raw_attributes[attr_type]
            stat_key = self.stat_key_fn(tree_key, attr_type)
            if update_stats:
                self.normalizer.update(stat_key, values.view(-1))
            raw = values.to(self.device)
            base_attrs[attr_type] = raw
            norm_attrs[attr_type] = self.normalizer.normalize(stat_key, raw.view(-1))
        return {"info": info, "base_attrs": base_attrs, "norm_attrs": norm_attrs}

    def get_payload(self, sample_key, image_channel, tree_spec, *, use_cache: bool):
        raise NotImplementedError
