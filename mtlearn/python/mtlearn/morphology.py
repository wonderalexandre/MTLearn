"""Stable morphology facade for mtlearn.

The current backend is mtlearn's native ``_mtlearn`` extension. Keeping user
code behind this module lets mtlearn swap morphology backends later without
changing notebooks and high-level Python APIs.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from ._backends import _mtlearn as _backend


class TreeType(str, Enum):
    """Public tree-type constants accepted by morphology and CFP layers."""

    MAX_TREE = "max-tree"
    MIN_TREE = "min-tree"
    TREE_OF_SHAPES = "tree-of-shapes"

    def __str__(self) -> str:
        return self.value


Tree = _backend.Tree
WeightedTree = _backend.Tree
WeightedMorphologicalTree = _backend.WeightedMorphologicalTree
Attribute = _backend.Attribute
AttributeType = _backend.AttributeType
AttributeGroup = _backend.AttributeGroup
AttributeFilters = _backend.AttributeFilters
NodeIdSpace = _backend.NodeIdSpace
ToSInterpolation = _backend.ToSInterpolation


def create_max_tree(image):
    """Build a max-tree from a 2D ``numpy.uint8`` image.

    Args:
        image: Contiguous or convertible 2D uint8 image.

    Returns:
        A ``WeightedMorphologicalTree`` facade backed by the native extension.
    """
    return _backend.create_max_tree(image)


def create_min_tree(image):
    """Build a min-tree from a 2D ``numpy.uint8`` image.

    Args:
        image: Contiguous or convertible 2D uint8 image.

    Returns:
        A ``WeightedMorphologicalTree`` facade backed by the native extension.
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
        A ``WeightedMorphologicalTree`` facade backed by the native extension.
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
        A ``WeightedMorphologicalTree`` facade.

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

    ``None`` maps to ``ToSInterpolation.SelfDual``.
    """
    return _backend.normalize_tos_interpolation(interpolation)


def compute_attributes(tree, attributes: Iterable):
    """Compute several scalar attributes or attribute groups for ``tree``.

    Args:
        tree: A tree returned by this module.
        attributes: Iterable containing ``AttributeType`` values,
            ``AttributeGroup`` values, or a mix of both.

    Returns:
        ``(attribute_index, values)`` where ``attribute_index`` maps attribute
        names to output columns and ``values`` is a float32 NumPy array with one
        row per node slot in the selected node-id space.
    """
    return _backend.compute_attributes(tree, attributes)


def compute_single_attribute(tree, attribute):
    """Compute one scalar attribute for ``tree``.

    Args:
        tree: A tree returned by this module.
        attribute: One ``AttributeType`` value.

    Returns:
        A 1D float32 NumPy array indexed by morphology-tree node slot.
    """
    return _backend.compute_single_attribute(tree, attribute)


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
    "Tree",
    "TreeType",
    "WeightedTree",
    "WeightedMorphologicalTree",
    "Attribute",
    "AttributeType",
    "AttributeGroup",
    "AttributeFilters",
    "NodeIdSpace",
    "ToSInterpolation",
    "create_max_tree",
    "create_min_tree",
    "create_tree_of_shapes",
    "build_tree",
    "normalize_tree_type",
    "normalize_tos_interpolation",
    "compute_attributes",
    "compute_single_attribute",
    "describe_attribute",
    "describe_all_attributes",
    "expand_attribute_group",
    "create_attribute_filter",
    "is_tree",
]
