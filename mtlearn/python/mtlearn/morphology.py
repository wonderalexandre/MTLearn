"""Morphology facade backed by the native ``_mtlearn`` extension."""

from __future__ import annotations

from enum import Enum

from ._backends import _mtlearn as _backend


class TreeType(str, Enum):
    """Public tree-type constants accepted by morphology and CFP layers."""

    MAX_TREE = "max-tree"
    MIN_TREE = "min-tree"
    TREE_OF_SHAPES = "tree-of-shapes"

    def __str__(self) -> str:
        return self.value


ValuedMorphologicalTree = _backend.ValuedMorphologicalTree
Attribute = _backend.Attribute
AttributeType = _backend.AttributeType
AttributeGroup = _backend.AttributeGroup
AttributeFilters = _backend.AttributeFilters
ExtinctionSelectionPolicy = _backend.ExtinctionSelectionPolicy
NodeIdSpace = _backend.NodeIdSpace
ToSInterpolation = _backend.ToSInterpolation


def create_max_tree(image):
    """Build a max-tree from a 2D ``numpy.uint8`` image.

    Args:
        image: Contiguous or convertible 2D uint8 image.

    Returns:
        A ``ValuedMorphologicalTree`` facade backed by the native extension.
    """
    return _backend.create_max_tree(image)


def create_min_tree(image):
    """Build a min-tree from a 2D ``numpy.uint8`` image.

    Args:
        image: Contiguous or convertible 2D uint8 image.

    Returns:
        A ``ValuedMorphologicalTree`` facade backed by the native extension.
    """
    return _backend.create_min_tree(image)


def create_tree_of_shapes(
    image,
    interpolation=None,
    infinity_seed_row: int = 0,
    infinity_seed_col: int = 0,
):
    """Build a tree of shapes from a 2D ``numpy.uint8`` image.

    Args:
        image: Contiguous or convertible 2D uint8 image.
        interpolation: Optional tree-of-shapes interpolation. Accepts
            ``ToSInterpolation`` values or string aliases such as
            ``"self-dual"``, ``"min4c-max8c"``, and ``"min8c-max4c"``.
        infinity_seed_row: Boundary infinity seed row used by the backend.
        infinity_seed_col: Boundary infinity seed column used by the backend.

    Returns:
        A ``ValuedMorphologicalTree`` facade backed by the native extension.
    """
    return _backend.create_tree_of_shapes(
        image,
        interpolation=interpolation,
        infinity_seed_row=infinity_seed_row,
        infinity_seed_col=infinity_seed_col,
    )


def build_tree(
    image,
    tree_type: str | TreeType,
    *,
    tos_interpolation=None,
    tos_infinity_seed_row: int = 0,
    tos_infinity_seed_col: int = 0,
):
    """Build the morphology tree selected by ``tree_type``.

    Args:
        image: Contiguous or convertible 2D uint8 image.
        tree_type: ``TreeType`` value or one of the accepted string aliases for
            max-tree, min-tree, or tree of shapes.
        tos_interpolation: Optional tree-of-shapes interpolation used only when
            ``tree_type`` selects ``TREE_OF_SHAPES``.
        tos_infinity_seed_row: Boundary infinity seed row for tree of shapes.
        tos_infinity_seed_col: Boundary infinity seed column for tree of shapes.

    Returns:
        A ``ValuedMorphologicalTree`` facade.

    Raises:
        ValueError: If ``tree_type`` or tree-of-shapes interpolation is unknown.
    """
    return _backend.build_tree(
        image,
        normalize_tree_type(tree_type),
        tos_interpolation=tos_interpolation,
        tos_infinity_seed_row=tos_infinity_seed_row,
        tos_infinity_seed_col=tos_infinity_seed_col,
    )


def normalize_tree_type(tree_type: str | TreeType) -> str:
    """Return the canonical string name for a public tree type.

    Canonical values are ``"max-tree"``, ``"min-tree"``, and
    ``"tree-of-shapes"``.
    """
    return _backend.normalize_tree_type(tree_type)


def normalize_tos_interpolation(interpolation=None):
    """Return the native tree-of-shapes interpolation enum value.

    ``None`` maps to ``ToSInterpolation.SELF_DUAL``.
    """
    return _backend.normalize_tos_interpolation(interpolation)


def compute(
    tree,
    attributes,
    *,
    output_space=NodeIdSpace.MORPHOLOGICAL_TREE,
    dtype=None,
):
    """Compute a scalar as a 1D array, or a group or iterable as (layout, values).

    Attributes and groups accept enum values or canonical symbolic names.
    The default dtype is float32; float64 is also supported.
    """
    return _backend.compute(tree, attributes, output_space=output_space, dtype=dtype)


def describe_attribute(attribute):
    """Return a human-readable description for one scalar attribute."""
    return _backend.describe_attribute(attribute)


def describe_all_attributes():
    """Return descriptions for all public scalar attributes keyed by name."""
    return _backend.describe_all_attributes()


def expand_attribute_group(group):
    """Expand one ``AttributeGroup`` into its scalar ``AttributeType`` members."""
    return _backend.expand_attribute_group(group)


def create_attribute_filter(tree):
    """Create an attribute-filter helper bound to ``tree``."""
    return _backend.create_attribute_filter(tree)


def is_tree(value) -> bool:
    """Return whether ``value`` is a native mtlearn morphology tree handle."""
    return _backend.is_tree(value)


__all__ = [
    "TreeType",
    "ValuedMorphologicalTree",
    "Attribute",
    "AttributeType",
    "AttributeGroup",
    "AttributeFilters",
    "ExtinctionSelectionPolicy",
    "NodeIdSpace",
    "ToSInterpolation",
    "create_max_tree",
    "create_min_tree",
    "create_tree_of_shapes",
    "build_tree",
    "normalize_tree_type",
    "normalize_tos_interpolation",
    "compute",
    "describe_attribute",
    "describe_all_attributes",
    "expand_attribute_group",
    "create_attribute_filter",
    "is_tree",
]
