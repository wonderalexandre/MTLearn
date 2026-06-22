"""Normalization helpers for CFP filter specs."""

from __future__ import annotations

import math
import numbers
import re
from typing import Any, Mapping

from .... import morphology
from ..._helpers import (
    normalize_attributes_spec,
    validate_attributes_for_tree_type,
)
from ..component_registries import (
    normalize_constraint_configs,
    normalize_regularizer_configs,
    normalize_scoring_model,
)
from .normalized_filter_spec import NormalizedFilterSpec


def enum_name(value: Any) -> str:
    """Return an enum name when available, otherwise a string representation."""
    return getattr(value, "name", str(value))


def normalize_nonnegative_scalar(value: Any, name: str) -> float:
    """Normalize and validate a non-negative finite scalar."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a non-negative finite scalar.")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a non-negative finite scalar.")
    return scalar


def normalize_positive_scalar(value: Any, name: str) -> float:
    """Normalize and validate a positive finite scalar."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive finite scalar.")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return scalar


def filter_spec_tree_key(
    tree_type,
    tos_interpolation,
    tos_infinity_seed_row,
    tos_infinity_seed_col,
) -> str:
    """Return the cache key for a morphology tree configuration."""
    interpolation_name = enum_name(tos_interpolation) if tos_interpolation is not None else "None"
    return f"{tree_type}|{interpolation_name}|{tos_infinity_seed_row}|{tos_infinity_seed_col}"


_FILTER_SPEC_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_filter_spec_name(value: Any, index: int, seen_names: set[str]) -> str:
    if value is None:
        name = f"spec_{index:03d}"
    else:
        if not isinstance(value, str):
            raise TypeError("filter spec name must be a string.")
        name = value.strip()
        if not name:
            raise ValueError("filter spec name must be non-empty.")
        if _FILTER_SPEC_NAME_RE.fullmatch(name) is None:
            raise ValueError(
                "filter spec name must start with a letter or underscore and contain only letters, digits, and underscores."
            )

    if name in seen_names:
        raise ValueError(f"duplicate filter spec name: {name!r}")
    seen_names.add(name)
    return name


def normalize_filter_specs(
    filter_specs,
    *,
    default_tos_interpolation,
    default_tos_infinity_seed_row: int,
    default_tos_infinity_seed_col: int,
    default_score_sharpness: float = 1.0,
) -> tuple[NormalizedFilterSpec, ...]:
    """Normalize user filter spec mappings into internal CFP specs."""
    if filter_specs is None:
        raise ValueError("filter_specs must contain at least one filter specification.")

    normalized = []
    seen_names = set()
    for index, raw_spec in enumerate(filter_specs):
        if not isinstance(raw_spec, Mapping):
            raise TypeError("Each filter spec must be a mapping.")
        if "tree_type" not in raw_spec:
            raise ValueError("Each filter spec must define tree_type.")
        if "attributes" not in raw_spec:
            raise ValueError("Each filter spec must define attributes.")
        if "output_mode" in raw_spec:
            raise ValueError("output_mode was removed. CFP now reconstructs fixed altitude residues.")
        if "valuation" in raw_spec:
            raise ValueError("valuation was removed. CFP now reconstructs fixed altitude residues.")
        if "beta_f" in raw_spec:
            raise ValueError("filter spec beta_f was renamed to score_sharpness.")

        spec_name = _normalize_filter_spec_name(raw_spec.get("name", None), index, seen_names)
        tree_type = morphology.normalize_tree_type(raw_spec["tree_type"])
        raw_attributes = raw_spec["attributes"]
        raw_group = tuple(raw_attributes) if isinstance(raw_attributes, (list, tuple)) else (raw_attributes,)
        if len(raw_group) < 1:
            raise ValueError("Each filter spec must contain at least one attribute.")

        attributes = normalize_attributes_spec([raw_group], tree_type)[0][0]
        validate_attributes_for_tree_type(attributes, tree_type)
        scoring_model = normalize_scoring_model(raw_spec.get("scoring", None), len(attributes))
        score_sharpness = normalize_positive_scalar(
            raw_spec.get("score_sharpness", default_score_sharpness),
            "score_sharpness",
        )

        constraint_configs = normalize_constraint_configs(raw_spec.get("constraints", None))
        preserve_root = bool(raw_spec.get("preserve_root", False)) or any(
            config["kind"] == "preserve_root" for config in constraint_configs
        )
        monotonicity_weight = normalize_nonnegative_scalar(
            raw_spec.get("monotonicity_weight", 0.0),
            "monotonicity_weight",
        )
        regularizer_configs = normalize_regularizer_configs(raw_spec.get("regularizers", None))

        spec_tos_interpolation = raw_spec.get("tos_interpolation", default_tos_interpolation)
        if tree_type == morphology.TreeType.TREE_OF_SHAPES.value:
            spec_tos_interpolation = morphology.normalize_tos_interpolation(spec_tos_interpolation)
        spec_tos_infinity_seed_row = int(raw_spec.get("tos_infinity_seed_row", default_tos_infinity_seed_row))
        spec_tos_infinity_seed_col = int(raw_spec.get("tos_infinity_seed_col", default_tos_infinity_seed_col))
        tree_key = filter_spec_tree_key(
            tree_type,
            spec_tos_interpolation,
            spec_tos_infinity_seed_row,
            spec_tos_infinity_seed_col,
        )

        normalized.append(
            NormalizedFilterSpec(
                index=index,
                key=spec_name,
                tree_type=tree_type,
                tree_key=tree_key,
                attributes=tuple(attributes),
                scoring_model=scoring_model,
                score_sharpness=score_sharpness,
                preserve_root=preserve_root,
                monotonicity_weight=monotonicity_weight,
                constraint_configs=constraint_configs,
                regularizer_configs=regularizer_configs,
                tos_interpolation=spec_tos_interpolation,
                tos_infinity_seed_row=spec_tos_infinity_seed_row,
                tos_infinity_seed_col=spec_tos_infinity_seed_col,
            )
        )

    if not normalized:
        raise ValueError("filter_specs must contain at least one filter specification.")
    return tuple(normalized)
