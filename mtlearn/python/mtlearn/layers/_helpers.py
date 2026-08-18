"""Shared utilities for connected-filter preprocessing layers.

This module holds the small, state-agnostic helpers used by the CFP layer
implementations:

- conversion of PyTorch image tensors to ``np.uint8`` arrays accepted by the
  morphology backend;
- morphology-tree construction through the public ``mtlearn.morphology``
  facade.

The connected filtering operation exposed by
``mtlearn.ConnectedFilterPreprocessingTreeTraversal`` is differentiable. Tree
construction and attribute computation are not differentiable, so the layers
use these helpers outside the autograd path and pass their own cache/state
dictionaries explicitly.
"""
from __future__ import annotations

from typing import Any, Tuple, Iterable, Optional

import numpy as np
import torch
from .. import morphology

from torch.utils.data import Dataset


class IndexedDatasetWrapper(Dataset):
    """Wrap a supervised dataset so each sample carries its stable index.

    CFP layers can use the index to build cache keys that remain stable across
    DataLoader batches. The wrapper expects samples shaped like ``(x, y)`` or
    ``(x, y, ...)`` and returns ``((x, idx + index_offset), y)``.

    ``index_offset`` lets different dataset splits use disjoint cache keys even
    when their local sample indices all start at zero.
    """

    def __init__(self, base_dataset, *, index_offset: int = 0):
        """Store the dataset that will be indexed by this wrapper."""
        self.base_dataset = base_dataset
        self.index_offset = int(index_offset)

    def __len__(self):
        """Return the number of samples in the wrapped dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """Return ``((x, idx), y)`` for the sample at ``idx``.

        Extra fields in the original sample are intentionally ignored because
        CFP caching only needs the input image, target, and stable index.
        """
        sample = self.base_dataset[idx]
        if isinstance(sample, (list, tuple)):
            # Common supervised-dataset shapes: (x, y) or (x, y, name).
            x = sample[0]
            y = sample[1]
        else:
            raise ValueError("Dataset samples must be tuples/lists containing at least (x, y).")
        return (x, idx + self.index_offset), y



# --------------------------- conversion helpers ---------------------------

def group_name(group: Iterable[Any]) -> str:
    """Return a stable display/key name for an attribute group.

    Enum-like attributes use their ``.name`` when available, producing names
    such as ``"AREA+GRAY_HEIGHT"``.
    """
    return "+".join([getattr(t, "name", str(t)) for t in group])


def _attribute_name(attr_type: Any) -> str:
    """Return the public enum name used in validation and filtering."""
    return getattr(attr_type, "name", str(attr_type))


def _is_tree_of_shapes(tree_type: Optional[str]) -> bool:
    """Return whether a tree type name selects the tree-of-shapes backend."""
    return tree_type is not None and morphology.normalize_tree_type(tree_type) == "tree-of-shapes"


def _expand_attribute_spec_entry(attr_type: Any, tree_type: Optional[str] = None) -> Tuple[Any, ...]:
    """Expand backend attribute groups into scalar attributes for CFP layers."""
    if isinstance(attr_type, morphology.AttributeGroup):
        expanded = tuple(morphology.expand_attribute_group(attr_type))
        if _is_tree_of_shapes(tree_type) and _attribute_name(attr_type) in {"SHAPE", "ALL"}:
            return tuple(attr for attr in expanded if _attribute_name(attr) != "MAX_DIST")
        return expanded
    return (attr_type,)


def normalize_attributes_spec(
    attributes_spec: Iterable[Any],
    tree_type: Optional[str] = None,
) -> Tuple[list[Tuple[Any, ...]], list[Any]]:
    """Normalize CFP layer attribute groups into scalar backend attributes.

    ``attributes_spec`` groups control learnable CFP projections. A public
    ``morphology.AttributeGroup`` is therefore expanded in place to the scalar
    attributes it represents before caching and weight construction. For
    tree-of-shapes, group-provided ``MAX_DIST`` is omitted from ``SHAPE`` and
    ``ALL`` because that scalar attribute is undefined on ToS.
    """
    group_defs = []
    all_attr_types_set = set()
    for item in attributes_spec:
        raw_group = tuple(item) if isinstance(item, (list, tuple)) else (item,)
        expanded_group = []
        for attr_type in raw_group:
            expanded_group.extend(_expand_attribute_spec_entry(attr_type, tree_type))
        if len(expanded_group) < 1:
            raise ValueError("Each attribute group must contain at least one attribute.")
        group = tuple(expanded_group)
        group_defs.append(group)
        for attr_type in group:
            all_attr_types_set.add(attr_type)
    return group_defs, list(all_attr_types_set)


def to_numpy_u8(img2d_t: torch.Tensor) -> np.ndarray:
    """Convert a 2D tensor to a contiguous ``np.uint8`` image.

    Rules:
      - if the maximum value is <= 1.5, the tensor is treated as normalized
        image data in ``[0, 1]`` and scaled by 255;
      - otherwise, values are cast directly to ``uint8``.

    Values outside the ``uint8`` range may be truncated by PyTorch's cast.
    """
    t = img2d_t.detach().to("cpu")
    if t.dtype == torch.uint8:
        return (t if t.is_contiguous() else t.contiguous()).numpy()
    if t.numel() == 0:
        return t.to(torch.uint8).numpy()
    mx = float(t.max())
    if mx <= 1.5:
        u8 = (t * 255.0).to(torch.uint8)
    else:
        u8 = t.to(torch.uint8)
    return (u8 if u8.is_contiguous() else u8.contiguous()).numpy()


# ----------------------------- morphology trees ------------------------------

def build_tree(
    img_np: np.ndarray,
    tree_type: str,
    *,
    tos_interpolation=None,
    tos_infinity_seed_row: int = 0,
    tos_infinity_seed_col: int = 0,
):
    """Build the morphology tree requested by ``tree_type``.

    Args:
        img_np: 2D ``np.uint8`` image.
        tree_type: ``"max-tree"``, ``"min-tree"``, ``"tree-of-shapes"``, or
            the legacy ``"tos"`` alias.
    """
    return morphology.build_tree(
        img_np,
        tree_type,
        tos_interpolation=tos_interpolation,
        tos_infinity_seed_row=tos_infinity_seed_row,
        tos_infinity_seed_col=tos_infinity_seed_col,
    )


def validate_attributes_for_tree_type(attributes: Iterable[Any], tree_type: str) -> None:
    """Reject attribute requests that the selected tree type cannot compute."""
    if morphology.normalize_tree_type(tree_type) != "tree-of-shapes":
        return

    unsupported = []
    for attr_type in attributes:
        name = _attribute_name(attr_type)
        expanded = _expand_attribute_spec_entry(attr_type, tree_type)
        unsupported_members = [
            _attribute_name(scalar_attr)
            for scalar_attr in expanded
            if _attribute_name(scalar_attr) == "MAX_DIST"
        ]
        if unsupported_members and len(expanded) == 1:
            unsupported.append(name)
        elif unsupported_members:
            unsupported.append(f"{name} contains {', '.join(sorted(set(unsupported_members)))}")

    if unsupported:
        names = ", ".join(sorted(set(unsupported)))
        raise ValueError(
            "tree-of-shapes CFP does not support attributes that are undefined "
            f"for tree-of-shapes: {names}"
        )


__all__ = [
    "group_name",
    "normalize_attributes_spec",
    "to_numpy_u8",
    "build_tree",
    "validate_attributes_for_tree_type",
]
