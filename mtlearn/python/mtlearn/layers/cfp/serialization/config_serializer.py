"""Serialization helpers for CFP component configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..normalization import DEFAULT_SCALE_MODE


_FILTER_SPEC_CONFIG_KEYS = {
    "attributes",
    "constraints",
    "name",
    "regularizers",
    "score_sharpness",
    "scoring",
    "tos_infinity_seed_col",
    "tos_infinity_seed_row",
    "tos_interpolation",
    "tree_type",
}
_LAYER_CONFIG_KEYS = {
    "attribute_dtype",
    "clamp",
    "eps",
    "filter_specs",
    "clipped_zscore_floor",
    "clipped_zscore_radius",
    "in_channels",
    "scale_mode",
    "score_sharpness",
}


class ConfigSerializer:
    """Convert simple CFP components to and from config dictionaries."""

    schema_version = 1

    def __init__(self, registry=None):
        self.registry = registry

    def to_config(self, component) -> dict:
        """Return a shallow config dictionary for a component."""
        if hasattr(component, "to_config"):
            return component.to_config()
        config = {"kind": getattr(component, "kind", component.__class__.__name__)}
        for key, value in vars(component).items():
            if key.startswith("_"):
                continue
            config[key] = value
        return config

    def from_config(self, config: dict, **context):
        """Instantiate a component using the configured registry."""
        if self.registry is None:
            raise RuntimeError("ConfigSerializer.from_config requires a registry.")
        return self.registry.create(config, **context)

    @staticmethod
    def _enum_name(value: Any) -> str:
        return getattr(value, "name", str(value))

    def filter_spec_config(self, spec, *, include_training: bool = True) -> dict[str, Any]:
        """Return the serializable architecture config for one normalized spec."""
        tos_interpolation = None if spec.tos_interpolation is None else self._enum_name(spec.tos_interpolation)
        spec_config = {
            "name": spec.key,
            "tree_type": spec.tree_type,
            "attributes": [self._enum_name(attr) for attr in spec.attributes],
            "scoring": spec.scoring_model.to_config(),
            "score_sharpness": spec.score_sharpness,
            "tos_interpolation": tos_interpolation,
            "tos_infinity_seed_row": spec.tos_infinity_seed_row,
            "tos_infinity_seed_col": spec.tos_infinity_seed_col,
        }
        if spec.constraint_configs:
            spec_config["constraints"] = [dict(config) for config in spec.constraint_configs]
        if include_training:
            if spec.regularizer_configs:
                spec_config["regularizers"] = [dict(config) for config in spec.regularizer_configs]
        return spec_config

    def filter_spec_metadata(self, spec) -> dict[str, Any]:
        """Return inspection/export metadata for one normalized spec."""
        tos_interpolation = None if spec.tos_interpolation is None else self._enum_name(spec.tos_interpolation)
        return {
            "index": spec.index,
            "key": spec.key,
            "name": spec.key,
            "tree_type": spec.tree_type,
            "tree_key": spec.tree_key,
            "attributes": [self._enum_name(attr) for attr in spec.attributes],
            "scoring": spec.scoring_model.to_config(),
            "score_sharpness": spec.score_sharpness,
            "constraints": [dict(config) for config in spec.constraint_configs],
            "regularizers": [dict(config) for config in spec.regularizer_configs],
            "tos_interpolation": tos_interpolation,
            "tos_infinity_seed_row": spec.tos_infinity_seed_row,
            "tos_infinity_seed_col": spec.tos_infinity_seed_col,
        }

    def layer_config(self, layer) -> dict[str, Any]:
        """Return the architecture/configuration needed to reconstruct a CFP layer."""
        return {
            "in_channels": layer.in_channels,
            "filter_specs": [
                self.filter_spec_config(spec, include_training=True)
                for spec in layer.filter_specs
            ],
            "scale_mode": layer.scale_mode,
            "eps": layer.eps,
            "score_sharpness": layer.score_sharpness,
            "clamp": None if layer.clamp is None else list(layer.clamp),
            "clipped_zscore_radius": layer.clipped_zscore_radius,
            "clipped_zscore_floor": layer.clipped_zscore_floor,
            "attribute_dtype": layer.attribute_dtype.name,
        }

    def parameter_contract(self, layer) -> dict[str, Any]:
        """Return parameter names and shapes owned by a CFP layer."""
        scoring_models = {}
        for spec_name, scoring_model in layer._scoring_models.items():
            model_parameters = {
                name: list(parameter.shape)
                for name, parameter in scoring_model.named_parameters()
            }
            if model_parameters:
                scoring_models[spec_name] = model_parameters
        return {
            "weights": {name: list(parameter.shape) for name, parameter in layer._weights.items()},
            "biases": {name: list(parameter.shape) for name, parameter in layer._biases.items()},
            "scoring_models": scoring_models,
        }

    def inference_contract(self, layer) -> dict[str, Any]:
        """Return the CFP contract that defines forward/inference semantics."""
        return {
            "in_channels": layer.in_channels,
            "filter_specs": [
                self.filter_spec_config(spec, include_training=False)
                for spec in layer.filter_specs
            ],
            "scale_mode": layer.scale_mode,
            "eps": layer.eps,
            "score_sharpness": layer.score_sharpness,
            "clamp": None if layer.clamp is None else list(layer.clamp),
            "clipped_zscore_radius": layer.clipped_zscore_radius,
            "clipped_zscore_floor": layer.clipped_zscore_floor,
        }

    @staticmethod
    def training_contract_for_spec(spec) -> dict[str, Any]:
        """Return training-only settings for one normalized spec."""
        contract = {
            "name": spec.key,
        }
        if spec.regularizer_configs:
            contract["regularizers"] = [dict(config) for config in spec.regularizer_configs]
        return contract

    def training_contract(self, layer) -> dict[str, Any]:
        """Return training-only CFP settings such as regularization weights."""
        return {
            "filter_specs": [
                self.training_contract_for_spec(spec)
                for spec in layer.filter_specs
            ],
        }

    def contracts(self, layer) -> dict[str, Any]:
        """Return named CFP contracts for parameters, inference, and training."""
        return {
            "parameter_contract": self.parameter_contract(layer),
            "inference_contract": self.inference_contract(layer),
            "training_contract": self.training_contract(layer),
        }

    def deserialize_filter_spec_config(
        self,
        spec: Mapping[str, Any],
        *,
        attribute_from_name,
        tos_interpolation_from_name,
    ) -> dict[str, Any]:
        """Deserialize one serialized CFP filter spec."""
        if "tree_type" not in spec:
            raise ValueError("serialized filter spec is missing tree_type.")
        if "attributes" not in spec:
            raise ValueError("serialized filter spec is missing attributes.")
        unknown_keys = set(spec) - _FILTER_SPEC_CONFIG_KEYS
        if unknown_keys:
            names = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ValueError(f"unsupported serialized filter spec key(s): {names}")

        restored = {
            "tree_type": spec["tree_type"],
            "attributes": tuple(attribute_from_name(attr) for attr in spec["attributes"]),
        }
        if "name" in spec:
            restored["name"] = spec["name"]
        if "scoring" in spec:
            restored["scoring"] = spec["scoring"]
        if "score_sharpness" in spec:
            restored["score_sharpness"] = float(spec["score_sharpness"])
        if "constraints" in spec:
            restored["constraints"] = spec["constraints"]
        if "regularizers" in spec:
            restored["regularizers"] = spec["regularizers"]
        tos_interpolation = spec.get("tos_interpolation", None)
        if tos_interpolation is not None:
            restored["tos_interpolation"] = tos_interpolation_from_name(tos_interpolation)
        if "tos_infinity_seed_row" in spec:
            restored["tos_infinity_seed_row"] = int(spec["tos_infinity_seed_row"])
        if "tos_infinity_seed_col" in spec:
            restored["tos_infinity_seed_col"] = int(spec["tos_infinity_seed_col"])
        return restored

    def deserialize_layer_config(
        self,
        config: Mapping[str, Any],
        *,
        attribute_from_name,
        tos_interpolation_from_name,
    ) -> dict[str, Any]:
        """Deserialize a CFP layer config into constructor kwargs."""
        if not isinstance(config, Mapping):
            raise TypeError("ConnectedFilterPreprocessingLayer config must be a mapping.")
        if "config" in config and "filter_specs" not in config:
            config = config["config"]
        unknown_keys = set(config) - _LAYER_CONFIG_KEYS
        if unknown_keys:
            names = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ValueError(f"unsupported serialized layer config key(s): {names}")

        return {
            "in_channels": int(config["in_channels"]),
            "filter_specs": [
                self.deserialize_filter_spec_config(
                    spec,
                    attribute_from_name=attribute_from_name,
                    tos_interpolation_from_name=tos_interpolation_from_name,
                )
                for spec in config["filter_specs"]
            ],
            "scale_mode": config.get("scale_mode", DEFAULT_SCALE_MODE),
            "eps": float(config.get("eps", 1e-6)),
            "score_sharpness": float(config.get("score_sharpness", 1.0)),
            "clamp": config.get("clamp", None),
            "clipped_zscore_radius": float(config.get("clipped_zscore_radius", 3.0)),
            "clipped_zscore_floor": float(config.get("clipped_zscore_floor", 0.05)),
            "attribute_dtype": config.get("attribute_dtype", None),
        }
