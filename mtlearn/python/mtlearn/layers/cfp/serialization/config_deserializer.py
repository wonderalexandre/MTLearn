"""Deserialization helpers for CFP layer configs."""

from __future__ import annotations

from typing import Any, Mapping

from .... import morphology
from ..valuation import CFPValuation
from .config_serializer import ConfigSerializer
from ..specs.filter_spec_normalizer import normalize_nonnegative_scalar


class ConfigDeserializer:
    """Convert serialized CFP layer configs into constructor kwargs."""

    def __init__(self, serializer: ConfigSerializer | None = None):
        self.serializer = ConfigSerializer() if serializer is None else serializer

    @staticmethod
    def attribute_from_name(value: Any) -> Any:
        """Resolve serialized attribute/group names to morphology enums."""
        if not isinstance(value, str):
            return value
        for enum_type in (morphology.AttributeType, morphology.AttributeGroup):
            try:
                return getattr(enum_type, value)
            except AttributeError:
                pass
        raise ValueError(f"unknown CFP attribute or group name: {value}")

    @staticmethod
    def tos_interpolation_from_name(value: Any) -> Any:
        """Resolve serialized tree-of-shapes interpolation names."""
        if value is None or not isinstance(value, str):
            return value
        try:
            return getattr(morphology.ToSInterpolation, value)
        except AttributeError as exc:
            raise ValueError(f"unknown tree-of-shapes interpolation name: {value}") from exc

    @classmethod
    def valuation_from_config(cls, value: Any) -> CFPValuation:
        """Resolve serialized valuation configs to CFP valuation objects."""
        if value is None:
            return CFPValuation.ALTITUDE
        if isinstance(value, CFPValuation):
            return value
        if isinstance(value, str):
            if value == "altitude":
                return CFPValuation.ALTITUDE
            if value == "altitude_tophat":
                return CFPValuation.ALTITUDE_TOPHAT
            return CFPValuation.node_attribute(cls.attribute_from_name(value))
        if not isinstance(value, Mapping):
            return CFPValuation.node_attribute(cls.attribute_from_name(value))

        kind = value.get("kind", "altitude")
        if kind == "altitude":
            return CFPValuation.ALTITUDE
        if kind == "altitude_tophat":
            return CFPValuation.ALTITUDE_TOPHAT
        if kind == "node_attribute":
            return CFPValuation.node_attribute(cls.attribute_from_name(value.get("attribute")))
        raise ValueError(f"unknown CFP valuation kind in config: {kind!r}")

    def deserialize_filter_spec_config(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """Deserialize one serialized CFP filter spec."""
        return self.serializer.deserialize_filter_spec_config(
            spec,
            attribute_from_name=self.attribute_from_name,
            valuation_from_config=self.valuation_from_config,
            tos_interpolation_from_name=self.tos_interpolation_from_name,
            normalize_nonnegative_scalar=normalize_nonnegative_scalar,
        )

    def deserialize_layer_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Deserialize a full CFP layer config."""
        return self.serializer.deserialize_layer_config(
            config,
            attribute_from_name=self.attribute_from_name,
            valuation_from_config=self.valuation_from_config,
            tos_interpolation_from_name=self.tos_interpolation_from_name,
            normalize_nonnegative_scalar=normalize_nonnegative_scalar,
        )

    def canonical_contract(self, config: Mapping[str, Any], *, layer_cls) -> dict[str, Any]:
        """Return the canonical inference contract represented by ``config``."""
        return layer_cls(**self.deserialize_layer_config(config)).get_weight_contract()
