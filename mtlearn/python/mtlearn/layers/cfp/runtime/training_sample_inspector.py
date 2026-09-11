"""Training-sample inspection for CFP layers."""

from __future__ import annotations

import torch

from ..._helpers import to_numpy_u8
from ..preparation import PreparedBatch
from .forward_executor import ForwardExecutor


class TrainingSampleInspector:
    """Inspect cached or direct CFP tree payloads for one sample."""

    def inspect(
        self,
        layer,
        img: torch.Tensor,
        *,
        channel: int = 0,
        idx: int | None = None,
        build_if_missing: bool = True,
    ):
        """Return cached or direct attributes, altitude increments, and parameters per spec."""
        if isinstance(img, PreparedBatch):
            if idx is not None or img.shape[0] != 1:
                raise ValueError("Use inspect_prepared_sample with a batch index for prepared batches.")
            return self.inspect_prepared(layer, img, channel=channel)
        if img.dim() == 2:
            img_chw = img.unsqueeze(0)
        elif img.dim() == 3:
            img_chw = img
        else:
            raise ValueError(f"img must be (H, W) or (C, H, W); got {tuple(img.shape)}")

        channels, _, _ = img_chw.shape
        if channels != layer.in_channels and channels != 1:
            raise AssertionError(f"in_channels={layer.in_channels}, input C={channels}")
        channel_index = channel if channels > 1 else 0

        payloads = {}
        if idx is not None:
            base_key = f"{idx}_{channel_index}"
            for spec in layer.filter_specs:
                if build_if_missing:
                    layer._ensure_tree_payload_cached(base_key, img_chw[channel_index], spec.tree_key)
                elif not layer._tree_payload_cache.has(base_key, spec.tree_key):
                    raise KeyError("Tree/attributes not found in cache. Use build_if_missing=True.")
            layer._maybe_refresh_norm_for_key(base_key)
            payloads = dict(layer._tree_payload_cache.sample_payloads(base_key))
        else:
            img_np = to_numpy_u8(img_chw[channel_index].detach())
            for tree_key in layer._tree_spec_by_key:
                payloads[tree_key] = layer._tree_payload_provider.compute_payload(
                    img_np,
                    tree_key,
                    update_stats=False,
                )

        return self._describe(layer, payloads)

    def inspect_prepared(self, layer, batch, *, batch_index=0, channel=0):
        if not isinstance(batch, PreparedBatch):
            raise TypeError("inspect_prepared_sample expects a PreparedBatch.")
        ForwardExecutor._input(layer, batch)
        if not 0 <= batch_index < batch.shape[0] or not 0 <= channel < batch.shape[1]:
            raise IndexError("Prepared sample or channel index out of range.")
        payloads = {key: layer._tree_payload_provider.consume_prepared(
            batch.samples[batch_index][channel][key]) for key in layer._tree_spec_by_key}
        return self._describe(layer, payloads)

    @staticmethod
    def _describe(layer, payloads):
        specs = {}
        for spec in layer.filter_specs:
            payload = payloads[spec.tree_key]
            cols_raw = [
                payload["base_attrs"][attr_type].view(-1, 1)
                for attr_type in spec.attributes
            ]
            cols_norm = [
                payload["norm_attrs"][attr_type].view(-1, 1)
                for attr_type in spec.attributes
            ]
            spec_payload = {
                "tree_type": spec.tree_type,
                "attributes": spec.attributes,
                "scoring_model": layer._scoring_models[spec.key],
                "score_sharpness": spec.score_sharpness,
                "base_attrs": torch.cat(cols_raw, dim=1),
                "norm_attrs": torch.cat(cols_norm, dim=1),
                "altitude_increments": payload["info"]["residues"],
            }
            if spec.key in layer._weights:
                spec_payload["weight"] = layer._weights[spec.key]
                spec_payload["bias"] = layer._biases[spec.key]
            specs[spec.key] = spec_payload
        return {"specs": specs}
