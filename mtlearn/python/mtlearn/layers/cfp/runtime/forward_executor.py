"""Forward and regularization execution for CFP layers."""

from __future__ import annotations

import torch

from ..preparation import PreparedBatch
from ._prepared_lifetime import keep_prepared_alive


class ForwardExecutor:
    """Run CFP forward and training-regularization loops."""

    def forward(self, layer, x: torch.Tensor) -> torch.Tensor:
        """Apply all filter specs and return ``(B, C * specs, H, W)``."""
        prepared = isinstance(x, PreparedBatch)
        batch, shape = self._input(layer, x)
        batch_size, channels, height, width = shape

        out_dtype = layer._module_dtype()
        out = torch.empty(
            (batch_size, layer.out_channels, height, width),
            dtype=out_dtype,
            device=layer.device,
        )
        for batch_index in range(batch_size):
            for channel_index in range(channels):
                base_key = (batch.sample_key(batch_index, channel_index) if prepared else
                            f"{int(batch.index[batch_index])}_{channel_index}")
                direct_payloads = {}
                for spec in layer.filter_specs:
                    info, raw_attrs, norm_attrs = self._payload(
                        layer, batch, batch_index, channel_index, base_key, spec, direct_payloads,
                    )

                    score_sharpness = layer._score_sharpness_for_spec(spec)
                    layer._active_context = layer._context_for(
                        base_key,
                        batch_index,
                        channel_index,
                        spec,
                        mode="forward",
                        image_shape=(height, width),
                        score_sharpness=score_sharpness,
                        raw_attrs=raw_attrs,
                        norm_attrs=norm_attrs,
                    )
                    try:
                        y_out = layer._apply_spec(
                            spec,
                            info,
                            norm_attrs,
                            score_sharpness,
                        )
                    finally:
                        layer._active_context = None
                    output_channel = channel_index * layer.num_specs + spec.index
                    out[batch_index, output_channel].copy_(y_out, non_blocking=True)
        return keep_prepared_alive(out, batch) if prepared else out

    def regularization_penalty(self, layer, x: torch.Tensor) -> torch.Tensor:
        """Return the per-spec training regularization penalty."""
        prepared = isinstance(x, PreparedBatch)
        batch, shape = self._input(layer, x)
        batch_size, channels, height, width = shape

        active_specs = [
            spec
            for spec in layer.filter_specs
            if len(layer._regularizers[spec.key]) > 0
        ]
        if not active_specs or batch_size * channels == 0:
            return layer._zero_parameter_penalty()

        penalty = layer._zero_parameter_penalty()
        for batch_index in range(batch_size):
            for channel_index in range(channels):
                base_key = (batch.sample_key(batch_index, channel_index) if prepared else
                            f"{int(batch.index[batch_index])}_{channel_index}")
                direct_payloads = {}
                for spec in active_specs:
                    info, raw_attrs, norm_attrs = self._payload(
                        layer, batch, batch_index, channel_index, base_key, spec, direct_payloads,
                    )
                    score_sharpness = layer._score_sharpness_for_spec(spec)
                    layer._active_context = layer._context_for(
                        base_key,
                        batch_index,
                        channel_index,
                        spec,
                        mode="regularization_penalty",
                        image_shape=(height, width),
                        score_sharpness=score_sharpness,
                        raw_attrs=raw_attrs,
                        norm_attrs=norm_attrs,
                    )
                    try:
                        penalty = penalty + layer._regularization_penalty_for_spec(
                            spec,
                            info,
                            norm_attrs,
                        )
                    finally:
                        layer._active_context = None
        penalty = penalty / float(batch_size * channels)
        return keep_prepared_alive(penalty, batch) if prepared else penalty

    @staticmethod
    def _input(layer, x):
        if isinstance(x, PreparedBatch):
            x.validate_for(layer)
            if layer.scale_mode != "none":
                layer._require_fixed_dataset_stats()
                if not layer._stats_frozen:
                    raise RuntimeError("Prepared consumption requires frozen statistics; call fit_stats or freeze_ds_stats.")
            return x, x.shape
        batch = layer._batch_input(x)
        shape = batch.tensor.shape
        assert batch.tensor.dim() == 4, f"expected (B, C, H, W), got {tuple(shape)}"
        assert shape[1] == layer.in_channels, f"in_channels={layer.in_channels}, input C={shape[1]}"
        return batch, shape

    @staticmethod
    def _payload(layer, batch, batch_index, channel_index, base_key, spec, payloads):
        if isinstance(batch, PreparedBatch):
            if spec.tree_key not in payloads:
                payloads[spec.tree_key] = layer._tree_payload_provider.consume_prepared(
                    batch.samples[batch_index][channel_index][spec.tree_key])
            payload = payloads[spec.tree_key]
            return payload["info"], payload["base_attrs"], payload["norm_attrs"]
        return layer._get_tree_payload_for_sample(
            base_key, batch.tensor[batch_index, channel_index], spec, payloads,
            use_cache=batch.use_cache,
        )
