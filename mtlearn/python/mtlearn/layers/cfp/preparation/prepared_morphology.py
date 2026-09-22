"""CPU morphology data, independent of scoring and normalization."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from ..specs import FeatureSpec, TreeSpec
from ._identity import FORMAT_VERSION

_INDEX_FIELDS = ("tpre", "tpost", "parent", "node_of_pixel")
_INFO_FIELDS = set(_INDEX_FIELDS) | {"residues", "num_rows", "num_cols", "tree_type"}


@dataclass(frozen=True, eq=False)
class PreparedMorphology:
    """One CPU image/channel/tree. Tensor buffers are read-only by contract.

    Mappings are copied and protected, while tensors are shared without copying.
    Consumers must not mutate those tensors. ``input_id`` is a logical label,
    not a content fingerprint or permission to reuse changed pixels.
    """

    tree_spec: TreeSpec
    feature_spec: FeatureSpec
    info: Mapping[str, Any]
    raw_attributes: Mapping[Any, torch.Tensor]
    input_id: str | None = None
    format_version: int = FORMAT_VERSION

    def __post_init__(self):
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))
        object.__setattr__(self, "raw_attributes", MappingProxyType(dict(self.raw_attributes)))
        self.validate()

    @property
    def tree_key(self) -> str:
        return self.tree_spec.cache_key()

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.info["num_rows"], self.info["num_cols"]

    @property
    def num_nodes(self) -> int:
        return self.info["residues"].numel()

    @property
    def nbytes(self) -> int:
        """Bytes of unique CPU storages, including storage behind any views."""
        storages = {}
        for value in (*self.info.values(), *self.raw_attributes.values()):
            if torch.is_tensor(value):
                storage = value.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        return sum(storages.values())

    def validate(self, *, full: bool = False) -> None:
        """Check the schema; optionally scan values and traversal consistency."""
        if self.format_version != FORMAT_VERSION or not isinstance(self.tree_spec, TreeSpec):
            raise ValueError("Unsupported prepared morphology format/tree specification.")
        if not isinstance(self.feature_spec, FeatureSpec) or self.feature_spec.normalization is not None:
            raise ValueError("Prepared morphology must contain raw features without normalization.")
        if set(self.info) != _INFO_FIELDS:
            raise ValueError("Prepared morphology has missing or unexpected tree fields.")
        if self.info["tree_type"] != self.tree_spec.tree_type:
            raise ValueError("Prepared morphology tree type does not match its specification.")
        for key in ("num_rows", "num_cols"):
            if type(self.info[key]) is not int or self.info[key] <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if self.input_id is not None and not isinstance(self.input_id, str):
            raise TypeError("input_id must be a string or None.")
        attributes = self.feature_spec.attributes
        if not isinstance(attributes, tuple):
            raise ValueError("Prepared attribute order must be an immutable tuple.")
        if len(set(attributes)) != len(attributes) or set(attributes) != set(self.raw_attributes):
            raise ValueError("Prepared attributes must match their ordered feature specification.")
        for key in ("residues", *_INDEX_FIELDS):
            tensor = self.info[key]
            dtype = (torch.float32 if key == "residues" else
                     getattr(torch, "uint32", torch.int64) if key == "node_of_pixel" else torch.int64)
            self._check_tensor(tensor, dtype=dtype, name=key)
            if tensor.ndim != 1:
                raise ValueError(f"{key} must be one-dimensional.")
        n = self.num_nodes
        if n == 0:
            raise ValueError("Prepared morphology requires at least one node.")
        for key in _INDEX_FIELDS:
            expected = self.info["num_rows"] * self.info["num_cols"] if key == "node_of_pixel" else n
            if self.info[key].numel() != expected:
                raise ValueError(f"{key} has the wrong length.")
        dtypes = set()
        for key, tensor in self.raw_attributes.items():
            self._check_tensor(tensor, name=str(key))
            if tensor.dtype not in (torch.float32, torch.float64) or tensor.shape != (n, 1):
                raise ValueError("Raw attributes must have shape (nodes, 1) and a supported floating dtype.")
            dtypes.add(tensor.dtype)
        if len(dtypes) != 1:
            raise ValueError("All prepared attributes must share a dtype.")
        if not full:
            return
        for tensor in (self.info["residues"], *self.raw_attributes.values()):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Prepared morphology contains non-finite values.")
        # PyTorch unsigned comparison kernels are not available on all devices.
        owners = self.info["node_of_pixel"].to(torch.int64)
        if bool(((owners < 0) | (owners >= n)).any()):
            raise ValueError("node_of_pixel references an invalid node.")
        pre, post, parent = (self.info[k] for k in ("tpre", "tpost", "parent"))
        if bool(((parent < -1) | (parent >= n)).any()):
            raise ValueError("parent references an invalid node.")
        if bool(((pre < 0) | (pre >= n) | (post <= pre) | (post > n)).any()):
            raise ValueError("Invalid compact preorder/subtree bounds.")
        if bool((torch.bincount(pre, minlength=n) != 1).any()):
            raise ValueError("Preorder indices must identify distinct nodes.")
        roots = (parent == -1) | (parent == torch.arange(n))
        if int(roots.sum()) != 1:
            raise ValueError("Prepared morphology must have one root.")
        if int(pre[roots].item()) != 0 or int(post[roots].item()) != n:
            raise ValueError("The root interval must cover the complete preorder.")
        children = (~roots).nonzero().flatten()
        if bool(((pre[parent[children]] >= pre[children]) | (post[parent[children]] < post[children])).any()):
            raise ValueError("Parent traversal intervals must contain their children.")

    @staticmethod
    def _check_tensor(value, *, name, dtype=None):
        if not torch.is_tensor(value) or value.device.type != "cpu" or value.requires_grad:
            raise ValueError(f"{name} must be a detached CPU tensor.")
        if not value.is_contiguous() or (dtype is not None and value.dtype != dtype):
            raise ValueError(f"{name} has an invalid layout or dtype.")
