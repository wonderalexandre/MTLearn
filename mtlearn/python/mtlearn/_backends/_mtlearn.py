"""Adapter from the public mtlearn morphology facade to mtlearn's native bindings."""

from __future__ import annotations

from typing import Iterable

from .._native import load_bindings


_bindings = load_bindings()


Tree = _bindings.WeightedMorphologicalTree
WeightedMorphologicalTree = _bindings.WeightedMorphologicalTree
Attribute = _bindings.Attribute
AttributeType = _bindings.Attribute.Type
AttributeGroup = _bindings.Attribute.Group
AttributeFilters = _bindings.AttributeFilters
NodeIdSpace = _bindings.NodeIdSpace
ToSInterpolation = _bindings.ToSInterpolation


_TREE_TYPE_ALIASES = {
    "max": "max-tree",
    "maxtree": "max-tree",
    "max-tree": "max-tree",
    "max_tree": "max-tree",
    "min": "min-tree",
    "mintree": "min-tree",
    "min-tree": "min-tree",
    "min_tree": "min-tree",
    "tos": "tree-of-shapes",
    "treeofshapes": "tree-of-shapes",
    "tree-of-shapes": "tree-of-shapes",
    "tree_of_shapes": "tree-of-shapes",
}


_TOS_INTERPOLATION_ALIASES = {
    "min4_max8": ToSInterpolation.MIN4_MAX8,
    "min8_max4": ToSInterpolation.MIN8_MAX4,
    "selfdual": ToSInterpolation.SELF_DUAL,
    "self-dual": ToSInterpolation.SELF_DUAL,
    "self_dual": ToSInterpolation.SELF_DUAL,
    "min4cmax8c": ToSInterpolation.MIN4_MAX8,
    "min4c-max8c": ToSInterpolation.MIN4_MAX8,
    "min4c_max8c": ToSInterpolation.MIN4_MAX8,
    "min8cmax4c": ToSInterpolation.MIN8_MAX4,
    "min8c-max4c": ToSInterpolation.MIN8_MAX4,
    "min8c_max4c": ToSInterpolation.MIN8_MAX4,
}


def normalize_tree_type(tree_type: str) -> str:
    if hasattr(tree_type, "value"):
        tree_type = tree_type.value
    key = str(tree_type).strip().lower()
    try:
        return _TREE_TYPE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted({"max-tree", "min-tree", "tree-of-shapes", "tos"}))
        raise ValueError(f"unknown tree_type: {tree_type!r}; expected one of {valid}") from exc


def normalize_tos_interpolation(interpolation=None):
    if interpolation is None:
        return ToSInterpolation.SELF_DUAL
    if isinstance(interpolation, ToSInterpolation):
        return interpolation
    key = str(interpolation).strip().lower()
    try:
        return _TOS_INTERPOLATION_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_TOS_INTERPOLATION_ALIASES))
        raise ValueError(f"unknown tree-of-shapes interpolation: {interpolation!r}; expected one of {valid}") from exc


def create_max_tree(image):
    return WeightedMorphologicalTree.create_max_tree(image)


def create_min_tree(image):
    return WeightedMorphologicalTree.create_min_tree(image)


def create_tree_of_shapes(
    image,
    interpolation=None,
    infinity_seed_row: int = 0,
    infinity_seed_col: int = 0,
):
    return WeightedMorphologicalTree.create_tree_of_shapes(
        image,
        normalize_tos_interpolation(interpolation),
        int(infinity_seed_row),
        int(infinity_seed_col),
    )


def build_tree(
    image,
    tree_type: str,
    *,
    tos_interpolation=None,
    tos_infinity_seed_row: int = 0,
    tos_infinity_seed_col: int = 0,
):
    canonical_tree_type = normalize_tree_type(tree_type)
    if canonical_tree_type == "max-tree":
        return create_max_tree(image)
    if canonical_tree_type == "min-tree":
        return create_min_tree(image)
    return create_tree_of_shapes(
        image,
        interpolation=tos_interpolation,
        infinity_seed_row=tos_infinity_seed_row,
        infinity_seed_col=tos_infinity_seed_col,
    )


def compute_attributes(
    tree,
    attributes: Iterable,
    output_space=NodeIdSpace.MORPHOLOGICAL_TREE,
    dtype=None,
):
    return Attribute.compute_attributes(tree, attributes, output_space, dtype)


def compute_single_attribute(
    tree,
    attribute,
    output_space=NodeIdSpace.MORPHOLOGICAL_TREE,
    dtype=None,
):
    return Attribute.compute_single_attribute(tree, attribute, output_space, dtype)


def describe_attribute(attribute):
    return Attribute.describe(attribute)


def describe_all_attributes():
    return Attribute.describe_all()


def expand_attribute_group(group):
    return Attribute.expand_group(group)


def create_attribute_filter(tree):
    return AttributeFilters(tree)


def is_tree(value) -> bool:
    return isinstance(value, Tree)
