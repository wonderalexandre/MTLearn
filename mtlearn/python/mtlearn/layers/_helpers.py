"""Shared utilities for connected-filter preprocessing layers.

This module holds the small, state-agnostic helpers used by the CFP layer
implementations:

- conversion of PyTorch image tensors to ``np.uint8`` arrays accepted by the
  morphology backend;
- morphology-tree construction through the public ``mtlearn.morphology``
  facade;
- dataset-statistics updates for supported normalization modes;
- cache refresh of normalized attributes when dataset statistics change.

The connected filtering operation exposed by
``mtlearn.ConnectedFilterPreprocessingTreeTraversal`` is differentiable. Tree
construction and attribute computation are not differentiable, so the layers
use these helpers outside the autograd path and pass their own cache/state
dictionaries explicitly.
"""
from __future__ import annotations

import pickle
from typing import Dict, Any, Tuple, Iterable, Mapping, Optional

import numpy as np
import torch
from .. import morphology

from torch.utils.data import Dataset


class IndexedDatasetWrapper(Dataset):
    """Wrap a supervised dataset so each sample carries its stable index.

    CFP layers can use the index to build cache keys that remain stable across
    DataLoader batches. The wrapper expects samples shaped like ``(x, y)`` or
    ``(x, y, ...)`` and returns ``((x, idx), y)``.
    """

    def __init__(self, base_dataset):
        """Store the dataset that will be indexed by this wrapper."""
        self.base_dataset = base_dataset

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
        return (x, idx), y



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


# ---------------------- dataset-statistics normalization ----------------------

def update_ds_stats(ds_stats: Dict[Any, Dict[str, torch.Tensor]],
                    scale_mode: str,
                    attr_type: Any,
                    a_raw_1d: torch.Tensor) -> bool:
    """Update dataset-level statistics for one raw attribute vector.

    Returns ``True`` when the stored statistics changed. Supported modes:

    - ``minmax01`` expands the stored ``[amin, amax]`` range as samples arrive;
    - ``zscore_tree`` accumulates ``count``, ``sum``, and ``sumsq``;
    - ``none`` leaves the statistics unchanged.
    """
    if scale_mode == "minmax01":
        amin_new = torch.min(a_raw_1d.detach())
        amax_new = torch.max(a_raw_1d.detach())
        changed = False
        if attr_type not in ds_stats:
            ds_stats[attr_type] = {"amin": amin_new, "amax": amax_new}
            changed = True
        else:
            st = ds_stats[attr_type]
            if amin_new < st["amin"]:
                st["amin"] = amin_new
                changed = True
            if amax_new > st["amax"]:
                st["amax"] = amax_new
                changed = True
        return changed
    elif scale_mode == "zscore_tree":
        v = a_raw_1d.detach().to(torch.float32)
        cnt = torch.tensor(v.numel(), dtype=torch.long)
        sm = torch.sum(v)
        sq = torch.sum(v * v)
        if attr_type not in ds_stats:
            ds_stats[attr_type] = {"count": cnt, "sum": sm, "sumsq": sq}
        else:
            ds_stats[attr_type]["count"] = ds_stats[attr_type]["count"] + cnt
            ds_stats[attr_type]["sum"]   = ds_stats[attr_type]["sum"] + sm
            ds_stats[attr_type]["sumsq"] = ds_stats[attr_type]["sumsq"] + sq
        return True
    elif scale_mode == "none":
        return False
    else:
        raise ValueError(f"unknown scale_mode: {scale_mode}")


def normalize_with_ds_stats(ds_stats: Mapping[Any, Dict[str, torch.Tensor]],
                            scale_mode: str,
                            eps: float,
                            attr_type: Any,
                            a_raw_1d: torch.Tensor) -> torch.Tensor:
    """Normalize a raw 1D attribute vector with dataset statistics.

    Supported modes:

    - ``minmax01``: ``(x - amin) / (amax - amin)``;
    - ``zscore_tree``: ``(x - mean) / std``;
    - ``none``: identity.
    """
    if scale_mode == "minmax01":
        stats = ds_stats.get(attr_type, None)
        if stats is None:
            amin = torch.min(a_raw_1d)
            amax = torch.max(a_raw_1d)
        else:
            amin = stats["amin"]
            amax = stats["amax"]
        denom = torch.clamp(amax - amin, min=eps)
        return (a_raw_1d - amin) / denom
    elif scale_mode == "zscore_tree":
        stats = ds_stats.get(attr_type, None)
        if stats is None or stats["count"].item() == 0:
            mean = torch.mean(a_raw_1d)
            std  = torch.std(a_raw_1d).clamp_min(eps)
        else:
            count = stats["count"].to(torch.float32)
            mean  = stats["sum"] / count
            var   = stats["sumsq"] / count - mean * mean
            std   = torch.sqrt(torch.clamp(var, min=eps))
        return (a_raw_1d - mean) / std
    elif scale_mode == "none":
        return a_raw_1d
    else:
        raise ValueError(f"unknown scale_mode: {scale_mode}")


def maybe_refresh_norm_for_key(key: str,
                               base_attrs: Dict[Any, Dict[Any, torch.Tensor]],
                               norm_attrs: Dict[Any, Dict[Any, torch.Tensor]],
                               all_attr_types: Iterable[Any],
                               ds_stats: Mapping[Any, Dict[str, torch.Tensor]],
                               scale_mode: str,
                               eps: float,
                               norm_epoch_by_key: Dict[str, int],
                               current_epoch: int) -> None:
    """Refresh normalized attributes for one cache key when stats advance."""
    last_epoch = norm_epoch_by_key.get(key, -1)
    if last_epoch == current_epoch:
        return
    per_attr_raw = base_attrs[key]
    per_attr_norm = {}
    for attr_type in all_attr_types:
        a_raw_1d = per_attr_raw[attr_type].squeeze(1)
        a_norm   = normalize_with_ds_stats(ds_stats, scale_mode, eps, attr_type, a_raw_1d)
        per_attr_norm[attr_type] = a_norm
    norm_attrs[key] = per_attr_norm
    norm_epoch_by_key[key] = current_epoch


# ------------------------- stats serialization helpers -------------------------

def _attribute_key(attr_type: Any) -> str:
    """Return the stable serialized key for an attribute enum value."""
    return getattr(attr_type, "name", str(attr_type))


def _attribute_from_key(key: Any) -> Any:
    """Convert a serialized attribute key back to the public enum value."""
    if not isinstance(key, str):
        # Legacy payloads loaded with weights_only=False may already contain
        # pybind enum instances as dictionary keys.
        return key
    try:
        return getattr(morphology.AttributeType, key)
    except AttributeError as exc:
        raise ValueError(f"unknown serialized attribute key: {key}") from exc


def serialize_ds_stats(ds_stats: Mapping[Any, Mapping[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Convert dataset stats to a torch-safe payload.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    Pybind enum keys are rejected by that safe loader, so new stats files store
    attribute names as plain strings and tensors as CPU tensors.
    """
    serialized = {}
    for attr_type, stats in ds_stats.items():
        serialized[_attribute_key(attr_type)] = {
            name: value.detach().cpu() if torch.is_tensor(value) else value
            for name, value in stats.items()
        }
    return serialized


def deserialize_ds_stats(serialized: Mapping[Any, Mapping[str, torch.Tensor]], device) -> Dict[Any, Dict[str, torch.Tensor]]:
    """Convert serialized stats back to enum-keyed dictionaries."""
    ds_stats = {}
    for key, stats in serialized.items():
        attr_type = _attribute_from_key(key)
        ds_stats[attr_type] = {
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in stats.items()
        }
    return ds_stats


def make_stats_payload(ds_stats: Mapping[Any, Mapping[str, torch.Tensor]], scale_mode: str) -> Dict[str, Any]:
    """Build the versioned payload saved by CFP ``save_stats`` methods."""
    return {
        "format_version": 2,
        "ds_stats": serialize_ds_stats(ds_stats),
        "scale_mode": scale_mode,
    }


def load_stats_payload(path: str, device, *, trusted_legacy_format: bool = False) -> Dict[str, Any]:
    """Load a CFP stats payload using PyTorch's safe loader when possible.

    Files written by current mtlearn versions use string keys and load with
    ``weights_only=True``. Older files may contain pybind enum keys and require
    ``trusted_legacy_format=True`` so callers opt into PyTorch's unsafe legacy
    pickle path explicitly.
    """
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        payload = torch.load(path, map_location=device)
    except pickle.UnpicklingError as exc:
        if not trusted_legacy_format:
            raise RuntimeError(
                "This stats file uses the legacy enum-keyed format. "
                "Re-save it with a current mtlearn version, or call "
                "load_stats(..., trusted_legacy_format=True) only for files "
                "you trust."
            ) from exc
        payload = torch.load(path, map_location=device, weights_only=False)

    return {
        "format_version": payload.get("format_version", 1),
        "scale_mode": payload.get("scale_mode"),
        "ds_stats": deserialize_ds_stats(payload.get("ds_stats", {}), device),
    }


__all__ = [
    "group_name",
    "normalize_attributes_spec",
    "to_numpy_u8",
    "build_tree",
    "validate_attributes_for_tree_type",
    "update_ds_stats",
    "normalize_with_ds_stats",
    "maybe_refresh_norm_for_key",
    "make_stats_payload",
    "load_stats_payload",
]
