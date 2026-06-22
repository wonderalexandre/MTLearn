"""Forward and regularization execution for CFP layers."""

from __future__ import annotations

import torch


class ForwardExecutor:
    """Run CFP forward and training-regularization loops."""

    def forward(self, layer, x: torch.Tensor) -> torch.Tensor:
        """Apply all filter specs and return ``(B, C * specs, H, W)``."""
        x, idx, use_cache = layer._batch_input(x)
        assert x.dim() == 4, f"expected (B, C, H, W), got {tuple(x.shape)}"
        batch_size, channels, height, width = x.shape
        assert channels == layer.in_channels, f"in_channels={layer.in_channels}, input C={channels}"

        out_dtype = layer._module_dtype()
        out = torch.empty(
            (batch_size, layer.out_channels, height, width),
            dtype=out_dtype,
            device=layer.device,
        )
        for batch_index in range(batch_size):
            for channel_index in range(channels):
                base_key = f"{int(idx[batch_index])}_{channel_index}"
                direct_payloads = {}
                for spec in layer.filter_specs:
                    info, norm_attrs = layer._get_tree_payload_for_sample(
                        base_key,
                        x[batch_index, channel_index],
                        spec,
                        direct_payloads,
                        use_cache=use_cache,
                    )

                    layer._active_context = layer._context_for(
                        base_key,
                        batch_index,
                        channel_index,
                        spec,
                        mode="forward",
                    )
                    try:
                        y_out = layer._apply_spec(
                            spec,
                            info,
                            norm_attrs,
                            layer._score_sharpness_for_spec(spec),
                        )
                    finally:
                        layer._active_context = None
                    output_channel = channel_index * layer.num_specs + spec.index
                    out[batch_index, output_channel].copy_(y_out, non_blocking=True)
        return out

    def regularization_penalty(self, layer, x: torch.Tensor) -> torch.Tensor:
        """Return the per-spec training regularization penalty."""
        x, idx, use_cache = layer._batch_input(x)
        assert x.dim() == 4, f"expected (B, C, H, W), got {tuple(x.shape)}"
        batch_size, channels, _, _ = x.shape
        assert channels == layer.in_channels, f"in_channels={layer.in_channels}, input C={channels}"

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
                base_key = f"{int(idx[batch_index])}_{channel_index}"
                direct_payloads = {}
                for spec in active_specs:
                    info, norm_attrs = layer._get_tree_payload_for_sample(
                        base_key,
                        x[batch_index, channel_index],
                        spec,
                        direct_payloads,
                        use_cache=use_cache,
                    )
                    layer._active_context = layer._context_for(
                        base_key,
                        batch_index,
                        channel_index,
                        spec,
                        mode="regularization_penalty",
                    )
                    try:
                        penalty = penalty + layer._regularization_penalty_for_spec(
                            spec,
                            info,
                            norm_attrs,
                        )
                    finally:
                        layer._active_context = None
        return penalty / float(batch_size * channels)
