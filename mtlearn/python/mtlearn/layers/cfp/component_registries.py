"""Default registries for CFP extension components."""

from __future__ import annotations

import numbers
from typing import Any, Mapping

from .constraints import PreserveRootConstraint
from .regularization import (
    AttributeOrderScoreMonotonicityRegularizer,
    EdgeScoreMonotonicityRegularizer,
    PathScoreMonotonicityRegularizer,
)
from .scoring import LinearSigmoidScorer, MLPScorer, ScoringModel
from .specs import SpecRegistry


def _normalize_hidden_units(value: Any) -> tuple[int, ...]:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        raise TypeError("MLP hidden_units must be an integer or a sequence of positive integers.")
    hidden_units = []
    for units in value:
        if isinstance(units, bool) or not isinstance(units, numbers.Integral):
            raise TypeError("MLP hidden_units must contain only positive integers.")
        units = int(units)
        if units < 1:
            raise ValueError("MLP hidden_units must contain only positive integers.")
        hidden_units.append(units)
    return tuple(hidden_units)


def _create_linear_sigmoid_scorer(*, num_features: int, **options) -> LinearSigmoidScorer:
    unsupported = set(options)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported linear_sigmoid scoring options: {names}")
    return LinearSigmoidScorer(num_features, owns_parameters=False)


def _create_mlp_scorer(
    *,
    num_features: int,
    hidden_units=None,
    activation: str = "relu",
    **options,
) -> MLPScorer:
    unsupported = set(options)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported mlp scoring options: {names}")
    normalized_hidden = _normalize_hidden_units(hidden_units if hidden_units is not None else (16,))
    return MLPScorer(
        num_features,
        hidden_units=normalized_hidden,
        activation=activation,
    )


SCORING_MODEL_REGISTRY = SpecRegistry()
SCORING_MODEL_REGISTRY.register(
    "linear_sigmoid",
    _create_linear_sigmoid_scorer,
)
SCORING_MODEL_REGISTRY.register("mlp", _create_mlp_scorer)


def normalize_scoring_model(value: Any, num_features: int) -> ScoringModel:
    """Normalize a scorer object or config mapping into a CFP scoring model."""
    if value is None:
        return LinearSigmoidScorer(num_features, owns_parameters=False)
    if isinstance(value, ScoringModel):
        expected = getattr(value, "num_features", num_features)
        if int(expected) != num_features:
            raise ValueError(f"scoring model expects {expected} features, got {num_features}.")
        return value
    if isinstance(value, Mapping):
        config = dict(value)
        config.setdefault("kind", "linear_sigmoid")
        try:
            return SCORING_MODEL_REGISTRY.create(config, num_features=num_features)
        except KeyError as exc:
            kind = config.get("kind")
            raise ValueError(f"unknown CFP scoring model kind: {kind!r}") from exc
    raise TypeError("scoring must be None, a ScoringModel, or a config mapping.")


def _create_preserve_root_constraint(**options) -> PreserveRootConstraint:
    unsupported = set(options)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported preserve_root constraint options: {names}")
    return PreserveRootConstraint()


def _create_edge_score_monotonicity_regularizer(
    *, weight: float = 1.0, **options
) -> EdgeScoreMonotonicityRegularizer:
    unsupported = set(options)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported edge_score_monotonicity regularizer options: {names}")
    return EdgeScoreMonotonicityRegularizer(weight=weight)


def _create_attribute_order_score_monotonicity_regularizer(
    *,
    weight: float = 1.0,
    feature_index: int = 0,
    direction: str = "increasing",
    min_gap: float = 0.0,
    **options,
) -> AttributeOrderScoreMonotonicityRegularizer:
    unsupported = set(options)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported attribute_order_score_monotonicity regularizer options: {names}")
    return AttributeOrderScoreMonotonicityRegularizer(
        weight=weight,
        feature_index=feature_index,
        direction=direction,
        min_gap=min_gap,
    )


def _create_path_score_monotonicity_regularizer(
    *,
    weight: float = 1.0,
    max_depth: int | None = None,
    **options,
) -> PathScoreMonotonicityRegularizer:
    unsupported = set(options)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported path_score_monotonicity regularizer options: {names}")
    return PathScoreMonotonicityRegularizer(weight=weight, max_depth=max_depth)


SCORE_CONSTRAINT_REGISTRY = SpecRegistry()
SCORE_CONSTRAINT_REGISTRY.register("preserve_root", _create_preserve_root_constraint)

REGULARIZER_REGISTRY = SpecRegistry()
REGULARIZER_REGISTRY.register(
    "edge_score_monotonicity",
    _create_edge_score_monotonicity_regularizer,
)
REGULARIZER_REGISTRY.register(
    "attribute_order_score_monotonicity",
    _create_attribute_order_score_monotonicity_regularizer,
)
REGULARIZER_REGISTRY.register(
    "path_score_monotonicity",
    _create_path_score_monotonicity_regularizer,
)


def _normalize_component_config_sequence(value: Any, component_name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{component_name}s must be a config mapping or a sequence of config mappings.")

    configs = []
    for item in value:
        if isinstance(item, str):
            config = {"kind": item}
        elif isinstance(item, Mapping):
            config = dict(item)
        else:
            raise TypeError(f"Each {component_name} config must be a mapping or kind string.")
        if "kind" not in config:
            raise ValueError(f"Each {component_name} config must define kind.")
        configs.append(config)
    return tuple(configs)


def normalize_constraint_configs(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate and normalize score-constraint config entries."""
    configs = _normalize_component_config_sequence(value, "constraint")
    for config in configs:
        kind = config["kind"]
        try:
            SCORE_CONSTRAINT_REGISTRY.resolve(kind)
        except KeyError as exc:
            raise ValueError(f"unknown CFP constraint kind: {kind!r}") from exc
    return configs


def normalize_regularizer_configs(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate and normalize regularizer config entries."""
    configs = _normalize_component_config_sequence(value, "regularizer")
    for config in configs:
        kind = config["kind"]
        try:
            REGULARIZER_REGISTRY.create(config)
        except KeyError as exc:
            raise ValueError(f"unknown CFP regularizer kind: {kind!r}") from exc
    return configs


def constraint_configs_for_spec(spec) -> tuple[dict[str, Any], ...]:
    """Return effective score-constraint configs for one normalized filter spec."""
    return tuple(spec.constraint_configs)


def regularizer_configs_for_spec(spec) -> tuple[dict[str, Any], ...]:
    """Return effective regularizer configs for one normalized filter spec."""
    return tuple(spec.regularizer_configs)


def create_score_constraint(config: Mapping[str, Any]):
    """Instantiate a registered score constraint."""
    return SCORE_CONSTRAINT_REGISTRY.create(config)


def create_regularizer(config: Mapping[str, Any]):
    """Instantiate a registered CFP regularizer."""
    return REGULARIZER_REGISTRY.create(config)
