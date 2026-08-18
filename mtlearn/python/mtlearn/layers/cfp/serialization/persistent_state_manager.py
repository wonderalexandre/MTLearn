"""Persistent state, checkpoint state, and parameter export helpers for CFP layers."""

from __future__ import annotations

from typing import Any, Mapping

import torch


class PersistentStateManager:
    """Manage persistent CFP state and inspection exports."""

    def save_stats(self, layer, path: str) -> None:
        """Save dataset-level normalization statistics for ``layer``."""
        payload = {
            "format_version": 3,
            "scale_mode": layer.scale_mode,
            "ds_stats": layer._serialize_ds_stats(),
        }
        torch.save(payload, path)

    def load_stats(self, layer, path: str, *, refresh_cache: bool = True) -> None:
        """Load dataset-level normalization statistics into ``layer``."""
        payload = torch.load(path, map_location=layer.device, weights_only=True)
        layer._ds_stats = layer._deserialize_ds_stats(payload.get("ds_stats", {}))
        layer._stats_epoch += 1
        if refresh_cache:
            layer.refresh_cached_normalization()

    def extra_state(self, layer) -> dict[str, Any]:
        """Return CFP state embedded by PyTorch ``state_dict``."""
        return {
            "inference_contract": layer.get_inference_contract(),
            "ds_stats": layer._serialize_ds_stats(),
            "stats_epoch": int(layer._stats_epoch),
            "stats_frozen": bool(layer._stats_frozen),
        }

    def set_extra_state(self, layer, state: Any) -> None:
        """Restore CFP state from PyTorch ``state_dict`` extra state."""
        if state is None:
            return
        if not isinstance(state, Mapping):
            raise TypeError("ConnectedFilterPreprocessingLayer extra state must be a mapping.")

        saved_contract = state.get("inference_contract", None)
        if saved_contract is None and "config" in state:
            saved_contract = state["config"]
        if saved_contract is not None and layer._canonical_contract(saved_contract) != layer.get_inference_contract():
            raise RuntimeError(
                "ConnectedFilterPreprocessingLayer checkpoint inference contract is incompatible "
                "with the current layer. Recreate the layer with ConnectedFilterPreprocessingLayer.from_config(...)."
            )

        layer._ds_stats = layer._deserialize_ds_stats(state.get("ds_stats", {}))
        layer._stats_epoch = int(state.get("stats_epoch", layer._stats_epoch + 1))
        layer._stats_frozen = bool(state.get("stats_frozen", layer._stats_frozen))
        layer.refresh_cached_normalization()

    def export_params(self, layer, path: str) -> None:
        """Export CFP parameters and metadata for inspection."""
        torch.save(self.parameter_payload(layer), path)

    @staticmethod
    def parameter_payload(layer) -> dict[str, Any]:
        """Return the payload written by ``export_params``."""
        return {
            "weights": {name: p.detach().cpu() for name, p in layer._weights.items()},
            "biases": {name: p.detach().cpu() for name, p in layer._biases.items()},
            "scoring_models": {
                name: {
                    param_name: param.detach().cpu()
                    for param_name, param in scoring_model.state_dict().items()
                }
                for name, scoring_model in layer._scoring_models.items()
                if scoring_model.state_dict()
            },
            "scale_mode": layer.scale_mode,
            "clamp": None if layer.clamp is None else list(layer.clamp),
            "config": layer.get_config(),
            "inference_contract": layer.get_inference_contract(),
            "contracts": layer.get_contracts(),
            "filter_specs": [layer._serialize_filter_spec(spec) for spec in layer.filter_specs],
        }
