"""Legacy connected-filter preprocessing layer.

This module preserves the former CFP constructor where ``attributes_spec``,
``tree_type``, and ``top_hat`` are global layer options. New experiments should
use ``ConnectedFilterPreprocessingLayer`` with per-output ``filter_specs``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import mtlearn
from .. import morphology
from .ConnectedFilterPreprocessingLayer import ConnectedFilterPreprocessingImplicitJacobianFunction
from ._helpers import (
    IndexedDatasetWrapper,
    build_tree,
    group_name,
    load_stats_payload,
    make_stats_payload,
    maybe_refresh_norm_for_key,
    normalize_attributes_spec,
    normalize_with_ds_stats,
    to_numpy_u8,
    update_ds_stats,
    validate_attributes_for_tree_type,
)


class ConnectedFilterPreprocessingLayerLegacy(torch.nn.Module):
    """Main learnable Connected Filter Preprocessing (CFP) layer.

    For each attribute group ``g`` with ``K`` normalized attributes
    ``A_g in R[num_nodes, K]``, the layer computes
    ``sigmoid(beta_f * (A_g @ w_g + b_g))`` as a node-wise filtering criterion.
    The criterion is applied to the tree residues and reconstructed to pixels
    through the implicit-Jacobian autograd function.

    This legacy implementation is kept for reproducing experiments that used
    the former global tree/output contract. It can run tensor operations on
    CUDA when ``device="cuda"`` while still building morphology trees through
    the CPU backend.

    Args:
        in_channels: Number of input channels.
        attributes_spec: Attribute groups. Each item is one group and must
            contain at least one morphology attribute enum.
        tree_type: ``"max-tree"``, ``"min-tree"``, ``"tree-of-shapes"``, or
            the legacy ``"tos"`` alias.
        device: Torch device used for parameters, cached tensors, and outputs.
        scale_mode: ``"minmax01"``, ``"zscore_tree"``, ``"hybrid"``, or
            ``"none"``.
        eps: Numerical floor for normalization denominators.
        beta_f: Forward sigmoid gain.
        top_hat: If true, output the tree-type-specific top-hat residual.
        clamp_logits: If true, clamp ``beta_f * logits`` to ``[-12, 12]``.
        hybrid_k: Number of standard deviations used for hybrid clipping.
        hybrid_floor_a: Lower bound used when remapping hybrid-normalized
            attributes to ``[a, 1]``.
        tos_interpolation: Tree-of-shapes interpolation policy. Accepts
            ``"self-dual"``, ``"min4c-max8c"``, ``"min8c-max4c"``, or the
            corresponding ``morphology.ToSInterpolation`` enum.
        tos_infinity_seed_row, tos_infinity_seed_col: Infinity seed used by
            the tree-of-shapes backend.
    """
    def __init__(self,
                 in_channels,
                 attributes_spec,
                 tree_type="max-tree",
                 device="cpu",
                 scale_mode: str = "hybrid",
                 eps: float = 1e-6,
                 beta_f: float = 1.0,
                 top_hat: bool = False,
                 clamp_logits: bool = False,
                 hybrid_k: float = 3.0,
                 hybrid_floor_a: float = 0.05,
                 tos_interpolation=None,
                 tos_infinity_seed_row: int = 0,
                 tos_infinity_seed_col: int = 0,
                 ):
        """Initialize CFP configuration, caches, and learnable parameters.

        The constructor normalizes the attribute specification into immutable
        groups, builds the flat attribute set used for cache construction, and
        creates one weight vector plus one scalar bias per group.
        """
        super().__init__()

        # Hybrid normalization configuration.
        self.hybrid_k = float(hybrid_k)
        self.hybrid_floor_a = float(hybrid_floor_a)

        self.in_channels = int(in_channels)
        self.tree_type   = morphology.normalize_tree_type(tree_type)
        self.device      = torch.device(device)
        self.scale_mode  = str(scale_mode)
        self.eps         = float(eps)
        self.beta_f      = float(beta_f)
        self.top_hat     = bool(top_hat)
        self.clamp_logits = bool(clamp_logits)
        if self.tree_type == "tree-of-shapes":
            self.tos_interpolation = morphology.normalize_tos_interpolation(tos_interpolation)
        else:
            self.tos_interpolation = tos_interpolation
        self.tos_infinity_seed_row = int(tos_infinity_seed_row)
        self.tos_infinity_seed_col = int(tos_infinity_seed_col)


        # Attribute groups and the flat set of scalar attribute types used by them.
        self.group_defs, self._all_attr_types = normalize_attributes_spec(attributes_spec, self.tree_type)
        validate_attributes_for_tree_type(self._all_attr_types, self.tree_type)

        self.num_groups   = len(self.group_defs)
        self.out_channels = self.in_channels * self.num_groups

        # Attribute, normalization, and implicit-Jacobian cache state.
        self._base_attrs = {}
        self._norm_attrs = {}
        self._stats_epoch = 0
        self._norm_epoch_by_key = {}
        self._ds_stats = {}
        self._stats_frozen = False
        self._info_jacobian = {}

        # Learnable parameters: one weight vector and one bias per group.
        self._weights = torch.nn.ParameterDict()
        self._biases  = torch.nn.ParameterDict()
        for g, group in enumerate(self.group_defs):
            k = len(group)
            gname = "+".join([t.name for t in group])
            w = torch.empty(k, dtype=torch.float32, device=self.device)
            b = torch.empty(1, dtype=torch.float32, device=self.device)
            # Xavier-like initialization for a one-dimensional parameter vector.
            fan_in, fan_out = k, 1
            gain = 1.0
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std
            torch.nn.init.uniform_(w, -a, a)
            torch.nn.init.constant_(b, 0.0)
            self._weights[gname] = torch.nn.Parameter(w, requires_grad=True)
            self._biases[gname]  = torch.nn.Parameter(b, requires_grad=True)

    # ---------- helpers ----------
    def _group_name(self, group):
        """Return the stable parameter/cache name for an attribute group."""
        return group_name(group)

    def _clamp_bounds(self):
        if self.clamp_logits:
            return -12.0, 12.0
        return None, None

    def _to_numpy_u8(self, img2d_t: torch.Tensor) -> np.ndarray:
        """Convert one image channel to the backend's ``np.uint8`` format."""
        return to_numpy_u8(img2d_t)

    def _compute_tree_info_for_jacobian(self, img_np: np.ndarray):
        """Build the morphology tree and implicit-Jacobian metadata.

        Returned metadata contains:

        - ``residues``: node residues;
        - ``tpre`` / ``tpost``: entry and exit times for each node;
        - ``parent``: parent index for each node;
        - ``node_of_pixel``: flattened-pixel to node mapping;
        - ``numRows`` / ``numCols``: image dimensions;
        - ``tree_type``: tree type used to build the structure;
        - ``order_forward`` / ``order_backward``: cached orders kept for API
          compatibility with the autograd function.

        No explicit mask is needed because the backend uses ``parent[root] =
        root``.
        """
        tree = build_tree(
            img_np,
            self.tree_type,
            tos_interpolation=self.tos_interpolation,
            tos_infinity_seed_row=self.tos_infinity_seed_row,
            tos_infinity_seed_col=self.tos_infinity_seed_col,
        )
        residues, tpre, tpost, parent, node_of_pixel = (
            mtlearn.ConnectedFilterPreprocessingTreeTensors.get_info_for_jacobian(tree)
        )
        info = {
            "residues": residues.to(self.device),
            "tpre": tpre.to(self.device),
            "tpost": tpost.to(self.device),
            "parent": parent.to(self.device),
            "node_of_pixel": node_of_pixel.to(self.device),
            "numRows": tree.numRows,
            "numCols": tree.numCols,
            "tree_type": self.tree_type,
        }
        # Cached forward order.
        info["order_forward"] = torch.argsort(tpre, descending=False).to(self.device)
        # Cached backward order.
        info["order_backward"] = torch.argsort(tpre).to(self.device)
        return tree, info

    # ---------- normalization with hybrid support ----------
    def _update_ds_stats(self, attr_type, a_raw_1d: torch.Tensor):
        """Update dataset statistics for one raw attribute vector."""
        if getattr(self, "_stats_frozen", False):
            return
        smode = self.scale_mode
        if smode == "hybrid":
            smode = "zscore_tree"
        changed = update_ds_stats(self._ds_stats, smode, attr_type, a_raw_1d)
        if changed:
            self._stats_epoch += 1

    def _normalize_with_ds_stats(self, attr_type, a_raw_1d: torch.Tensor) -> torch.Tensor:
        """Normalize a raw attribute vector according to ``scale_mode``.

        Hybrid mode applies three steps:

        1. z-score using dataset-level mean and standard deviation;
        2. clipping to ``[-hybrid_k, hybrid_k]``;
        3. remapping to ``[hybrid_floor_a, 1]``.
        """
        if self.scale_mode != "hybrid":
            return normalize_with_ds_stats(self._ds_stats, self.scale_mode, self.eps, attr_type, a_raw_1d)

        # Hybrid mode.
        st = self._ds_stats.get(attr_type, None)
        if st is None:
            # Before stats exist, preserve the raw values so early calls do not fail.
            return a_raw_1d

        # Dataset-level z-score statistics.
        count = st["count"].to(torch.float32)
        mean  = (st["sum"] / torch.clamp(count, min=1.0)) if count.item() > 0 else torch.tensor(0.0, device=a_raw_1d.device)
        var   = (st["sumsq"] / torch.clamp(count, min=1.0) - mean * mean) if count.item() > 0 else torch.tensor(0.0, device=a_raw_1d.device)
        std   = torch.sqrt(torch.clamp(var, min=self.eps))

        # 1) z-score
        x = (a_raw_1d - mean) / std
        # 2) clip em [-k, +k]
        k = torch.tensor(self.hybrid_k, dtype=x.dtype, device=x.device)
        x = torch.clamp(x, -k, k)
        # 3) rescale to [a, 1].
        a = torch.tensor(self.hybrid_floor_a, dtype=x.dtype, device=x.device)
        x01 = a + (1.0 - a) * ((x + k) / (2.0 * k))
        return x01

    def _maybe_refresh_norm_for_key(self, key: str):
        """Refresh normalized cached attributes if dataset stats changed."""
        # Nothing to refresh until raw attributes are cached.
        if key not in self._base_attrs:
            return

        # Cache is already current for this stats epoch.
        if self._norm_epoch_by_key.get(key, -1) == self._stats_epoch:
            return

        if self.scale_mode == "hybrid":
            # Reapply hybrid normalization attribute by attribute.
            per_attr_raw = self._base_attrs[key]           # dict[attr_type] -> (numNodes,1)
            per_attr_norm = {}
            for attr_type, a_raw_2d in per_attr_raw.items():
                a_raw_1d = a_raw_2d.view(-1)              # (numNodes,)
                a_norm   = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                per_attr_norm[attr_type] = a_norm
            self._norm_attrs[key] = per_attr_norm
            self._norm_epoch_by_key[key] = self._stats_epoch
        else:
            # Non-hybrid modes are shared across CFP implementations.
            maybe_refresh_norm_for_key(
                key,
                self._base_attrs,
                self._norm_attrs,
                self._all_attr_types,
                self._ds_stats,
                self.scale_mode,
                self.eps,
                self._norm_epoch_by_key,
                self._stats_epoch
            )

    def freeze_ds_stats(self):
        """Stop collecting dataset statistics for future samples."""
        self._stats_frozen = True

    def unfreeze_ds_stats(self):
        """Resume collecting dataset statistics for future samples."""
        self._stats_frozen = False

    def save_stats(self, path: str):
        """Save normalization statistics and scale mode for reproducibility."""
        payload = make_stats_payload(self._ds_stats, self.scale_mode)
        torch.save(payload, path)
        print(f"[ConnectedLinearUnit] stats saved to {path}")

    def load_stats(self, path: str, refresh_cache: bool = True, *, trusted_legacy_format: bool = False):
        """Load normalization statistics and optionally refresh cached attrs."""
        payload = load_stats_payload(path, self.device, trusted_legacy_format=trusted_legacy_format)
        self._ds_stats = payload.get("ds_stats", {})
        # Invalidate normalized values derived from older statistics.
        self._stats_epoch += 1
        if refresh_cache:
            self.refresh_cached_normalization()

    # ---------- tree and attribute construction ----------
    def _ensure_tree_info_and_attributes_cached(self, key: str, img_t: torch.Tensor):
        """Ensure tree metadata and raw/normalized attributes exist in cache."""
        if key in self._info_jacobian:
            return

        img_np = self._to_numpy_u8(img_t.detach())
        tree, info = self._compute_tree_info_for_jacobian(img_np)
        # The tree itself is not cached by the implicit implementation.
        self._info_jacobian[key] = info

        per_attr_raw, per_attr_norm = {}, {}
        for attr_type in self._all_attr_types:
            attr_np  = morphology.compute_attributes(tree, [attr_type])[1]
            a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
            self._update_ds_stats(attr_type, a_raw_1d)
            a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
            per_attr_raw[attr_type]  = a_raw_1d.unsqueeze(1)
            per_attr_norm[attr_type] = a_norm

        self._base_attrs[key] = per_attr_raw
        self._norm_attrs[key] = per_attr_norm
        self._norm_epoch_by_key[key] = self._stats_epoch

    # ---------- inspection ----------
    def inspect_training_sample(self, img: torch.Tensor, channel: int = 0, idx: int | None = None, build_if_missing: bool = True):
        """Return cached or on-the-fly inspection data for one sample.

        When ``idx`` is provided, the method inspects cached data under
        ``f"{idx}_{channel}"``. Without an index, the tree and attributes are
        computed on the fly and are not persisted.
        """
        # Normalize image layout to (C, H, W).
        if img.dim() == 2:
            imgCHW = img.unsqueeze(0)
        elif img.dim() == 3:
            imgCHW = img
        else:
            raise ValueError(f"img must be (H, W) or (C, H, W); got {tuple(img.shape)}")

        C, H, W = imgCHW.shape
        if C != self.in_channels:
            if C != 1:
                raise AssertionError(f"in_channels={self.in_channels}, input C={C}")

        c = channel if C > 1 else 0

        if idx is not None:
            key = f"{idx}_{c}"
            use_cache = True
        else:
            use_cache = False

        if use_cache:
            if (key not in self._info_jacobian) and build_if_missing:
                self._ensure_tree_info_and_attributes_cached(key, imgCHW[c])
            elif key not in self._info_jacobian:
                raise KeyError("Tree/attributes not found in cache. Use build_if_missing=True.")
            self._maybe_refresh_norm_for_key(key)
            # The implicit implementation does not store the tree object itself.
            tree = None
            base_attrs_by_group = {}
            norm_attrs_by_group = {}
            weights_by_group    = {}
            bias_by_group       = {}
            for group in self.group_defs:
                gname = self._group_name(group)
                cols_raw  = [self._base_attrs[key][attr_type].view(-1, 1) for attr_type in group]
                cols_norm = [self._norm_attrs[key][attr_type].view(-1, 1) for attr_type in group]
                A_raw  = torch.cat(cols_raw,  dim=1)
                A_norm = torch.cat(cols_norm, dim=1)
                base_attrs_by_group[gname] = A_raw
                norm_attrs_by_group[gname] = A_norm
                weights_by_group[gname]    = self._weights[gname]
                bias_by_group[gname]       = self._biases[gname]
        else:
            print("[inspect_training_sample] Running without cache; computing tree and attributes directly.")
            img_np = self._to_numpy_u8(imgCHW[c].detach())
            tree, info = self._compute_tree_info_for_jacobian(img_np)
            residues = info["residues"]
            base_attrs_by_group = {}
            norm_attrs_by_group = {}
            weights_by_group    = {}
            bias_by_group       = {}
            for group in self.group_defs:
                gname = self._group_name(group)
                cols_raw, cols_norm = [], []
                for attr_type in group:
                    attr_np = morphology.compute_attributes(tree, [attr_type])[1]
                    a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
                    a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                    cols_raw.append(a_raw_1d.unsqueeze(1))
                    cols_norm.append(a_norm.view(-1, 1))
                A_raw = torch.cat(cols_raw, dim=1)
                A_norm = torch.cat(cols_norm, dim=1)
                base_attrs_by_group[gname] = A_raw
                norm_attrs_by_group[gname] = A_norm
                weights_by_group[gname]    = self._weights[gname]
                bias_by_group[gname]       = self._biases[gname]

        return {
            "tree": tree,
            "base_attrs_by_group": base_attrs_by_group,
            "norm_attrs_by_group": norm_attrs_by_group,
            "weights_by_group": weights_by_group,
            "bias_by_group": bias_by_group,
        }

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CFP to a batch and return ``(B, C * groups, H, W)`` output.

        The input can be a tensor, ``(x, idx)``, or ``[x, idx]`` from a
        DataLoader. Indexed inputs use persistent caches keyed by sample index
        and channel; plain tensor inputs build trees and attributes on demand.
        """
        # Match the input conventions used by the CFP reference layers.
        if isinstance(x, tuple) and len(x) == 2:
            x, idx = x
            use_cache = True
        elif isinstance(x, list) and len(x) == 2 and isinstance(x[1], torch.Tensor) and x[1].dim() == 1:
            x, idx = x[0], x[1]
            use_cache = True
        else:
            if isinstance(x, list):
                x = torch.stack(x, dim=0)
            idx = torch.arange(x.size(0), device=x.device)
            use_cache = False

        assert x.dim() == 4, f"expected (B, C, H, W), got {tuple(x.shape)}"
        B, C, H, W = x.shape
        assert C == self.in_channels, f"in_channels={self.in_channels}, input C={C}"

        out_dtype = next(self.parameters()).dtype
        out = torch.empty((B, self.out_channels, H, W), dtype=out_dtype, device=self.device)
        clamp_min, clamp_max = self._clamp_bounds()

        for b in range(B):
            for c in range(C):
                if use_cache:
                    # Use idx as part of the persistent per-channel cache key.
                    key = f"{int(idx[b])}_{c}"
                    self._ensure_tree_info_and_attributes_cached(key, x[b, c])
                    self._maybe_refresh_norm_for_key(key)
                    info = self._info_jacobian[key]
                    for g, group in enumerate(self.group_defs):
                        gname = self._group_name(group)
                        # Build A_norm by stacking one normalized column per group attribute.
                        cols = [self._norm_attrs[key][attr_type].view(-1, 1).to(dtype=out_dtype) for attr_type in group]
                        A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                        y_ch = ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
                            self._weights[gname],
                            self._biases[gname],
                            info["residues"].to(dtype=out_dtype),
                            info["tpre"],
                            info["tpost"],
                            info["parent"],
                            info["node_of_pixel"],
                            A_norm,
                            info["numRows"],
                            info["numCols"],
                            self.beta_f,
                            clamp_min,
                            clamp_max,
                            info["order_forward"],
                            info["order_backward"]
                        )
                        if self.top_hat:
                            x_bc = x[b, c].to(dtype=out_dtype, device=self.device)
                            tt = self.tree_type
                            if tt == "max-tree":
                                y_out = x_bc - y_ch
                            elif tt == "min-tree":
                                y_out = y_ch - x_bc
                            else:
                                y_out = torch.abs(y_ch - x_bc)
                        else:
                            y_out = y_ch
                        out[b, c * self.num_groups + g].copy_(y_out, non_blocking=True)
                else:
                    # No persistent key was provided; build tree and attributes directly.
                    img_np = self._to_numpy_u8(x[b, c].detach())
                    tree, info = self._compute_tree_info_for_jacobian(img_np)
                    residues = info["residues"]
                    per_attr_norm = {}
                    for attr_type in self._all_attr_types:
                        attr_np = morphology.compute_attributes(tree, [attr_type])[1]
                        a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
                        a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                        per_attr_norm[attr_type] = a_norm
                    for g, group in enumerate(self.group_defs):
                        gname = self._group_name(group)
                        cols = [per_attr_norm[attr_type].view(-1, 1).to(dtype=out_dtype) for attr_type in group]
                        A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                        y_ch = ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
                            self._weights[gname],
                            self._biases[gname],
                            info["residues"].to(dtype=out_dtype),
                            info["tpre"],
                            info["tpost"],
                            info["parent"],
                            info["node_of_pixel"],
                            A_norm,
                            info["numRows"],
                            info["numCols"],
                            self.beta_f,
                            clamp_min,
                            clamp_max,
                            info["order_forward"],
                            info["order_backward"]
                        )
                        if self.top_hat:
                            x_bc = x[b, c].to(dtype=out_dtype, device=self.device)
                            tt = self.tree_type
                            if tt == "max-tree":
                                y_out = x_bc - y_ch
                            elif tt == "min-tree":
                                y_out = y_ch - x_bc
                            else:
                                y_out = torch.abs(y_ch - x_bc)
                        else:
                            y_out = y_ch
                        out[b, c * self.num_groups + g].copy_(y_out, non_blocking=True)

        return out

    # ---------- prediction / inference ----------
    def predict(self, x: torch.Tensor, beta_f: float = 1000.0) -> torch.Tensor:
        """Run inference with a caller-provided forward sigmoid gain."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            # Same input handling as forward.
            if isinstance(x, tuple) and len(x) == 2:
                x, idx = x
                use_cache = True
            elif isinstance(x, list) and len(x) == 2 and isinstance(x[1], torch.Tensor) and x[1].dim() == 1:
                x, idx = x[0], x[1]
                use_cache = True
            else:
                if isinstance(x, list):
                    x = torch.stack(x, dim=0)
                idx = torch.arange(x.size(0), device=x.device)
                use_cache = False

            B, C, H, W = x.shape
            out_dtype = next(self.parameters()).dtype
            out = torch.empty((B, self.out_channels, H, W), dtype=out_dtype, device=self.device)
            clamp_min, clamp_max = self._clamp_bounds()
            for b in range(B):
                for c in range(C):
                    if use_cache:
                        key = f"{int(idx[b])}_{c}"
                        self._ensure_tree_info_and_attributes_cached(key, x[b, c])
                        self._maybe_refresh_norm_for_key(key)
                        info = self._info_jacobian[key]
                        for g, group in enumerate(self.group_defs):
                            gname = self._group_name(group)
                            cols = [self._norm_attrs[key][attr_type].view(-1, 1).to(dtype=out_dtype) for attr_type in group]
                            A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                            y_ch = ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
                                self._weights[gname],
                                self._biases[gname],
                                info["residues"].to(dtype=out_dtype),
                                info["tpre"],
                                info["tpost"],
                                info["parent"],
                                info["node_of_pixel"],
                                A_norm,
                                info["numRows"],
                                info["numCols"],
                                beta_f,  # caller-provided beta_f
                                clamp_min,
                                clamp_max,
                                info["order_forward"],
                                info["order_backward"]
                            )
                            if self.top_hat:
                                x_bc = x[b, c].to(dtype=out_dtype, device=self.device)
                                tt = self.tree_type
                                if tt == "max-tree":
                                    y_out = x_bc - y_ch
                                elif tt == "min-tree":
                                    y_out = y_ch - x_bc
                                else:
                                    y_out = torch.abs(y_ch - x_bc)
                            else:
                                y_out = y_ch
                            out[b, c * self.num_groups + g].copy_(y_out, non_blocking=True)
                    else:
                        img_np = self._to_numpy_u8(x[b, c].detach())
                        tree, info = self._compute_tree_info_for_jacobian(img_np)
                        per_attr_norm = {}
                        for attr_type in self._all_attr_types:
                            attr_np = morphology.compute_attributes(tree, [attr_type])[1]
                            a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
                            a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                            per_attr_norm[attr_type] = a_norm
                        for g, group in enumerate(self.group_defs):
                            gname = self._group_name(group)
                            cols = [per_attr_norm[attr_type].view(-1, 1).to(dtype=out_dtype) for attr_type in group]
                            A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                            y_ch = ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
                                self._weights[gname],
                                self._biases[gname],
                                info["residues"].to(dtype=out_dtype),
                                info["tpre"],
                                info["tpost"],
                                info["parent"],
                                info["node_of_pixel"],
                                A_norm,
                                info["numRows"],
                                info["numCols"],
                                beta_f,  # caller-provided beta_f
                                clamp_min,
                                clamp_max,
                                info["order_forward"],
                                info["order_backward"]
                            )
                            if self.top_hat:
                                x_bc = x[b, c].to(dtype=out_dtype, device=self.device)
                                tt = self.tree_type
                                if tt == "max-tree":
                                    y_out = x_bc - y_ch
                                elif tt == "min-tree":
                                    y_out = y_ch - x_bc
                                else:
                                    y_out = torch.abs(y_ch - x_bc)
                            else:
                                y_out = y_ch
                            out[b, c * self.num_groups + g].copy_(y_out, non_blocking=True)
        if was_training:
            self.train()
        else:
            self.eval()
        return out

    # ---------- save / load ----------
    def save_params(self, path: str):
        """Save all group weights and biases."""
        payload = {
            "weights": { name: p.detach().cpu() for name, p in self._weights.items() },
            "biases":  { name: p.detach().cpu() for name, p in self._biases.items()  },
            "scale_mode": self.scale_mode,
        }
        torch.save(payload, path)
        print(f"[ConnectedLinearUnit] weights and biases saved to {path}")

    def get_params(self):
        """Return CPU clones of group weights and biases."""
        return (
            { name: p.detach().cpu().clone() for name, p in self._weights.items() },
            { name: p.detach().cpu().clone() for name, p in self._biases.items()  },
        )


    # ---------- cached-normalization utilities ----------
    def refresh_cached_normalization(self):
        """Recompute normalized attributes for every cached sample."""
        for key, per_attr_raw in self._base_attrs.items():
            per_attr_norm = {}
            for attr_type, a_raw_2d in per_attr_raw.items():
                a_raw_1d = a_raw_2d.view(-1)
                a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                per_attr_norm[attr_type] = a_norm
            self._norm_attrs[key] = per_attr_norm
            self._norm_epoch_by_key[key] = self._stats_epoch

    # ---------- initialization helpers ----------
    @staticmethod
    def _logit(p: float) -> float:
        """Return a numerically clipped logit for probability ``p``."""
        p = max(min(float(p), 1.0 - 1e-6), 1e-6)
        return math.log(p / (1.0 - p))

    @torch.no_grad()
    def init_identity_with_bias(self, p0: float = 0.995):
        """Initialize near identity by using only a positive bias.

        For each group, weights are set to zero and bias is set to
        ``logit(p0) / beta_f``. This keeps the initial filtering probability
        near ``p0`` while leaving trainable parameters free to move.
        """
        L = self._logit(p0) / float(self.beta_f)
        for group in self.group_defs:
            gname = self._group_name(group)
            self._weights[gname].zero_()
            self._biases[gname].fill_(L)

    @torch.no_grad()
    def init_identity_bias_zero(self, p0: float = 0.99):
        """Initialize near identity with zero bias under hybrid normalization.

        This assumes normalized attributes live in ``[a, 1]`` where
        ``a = hybrid_floor_a``. Each group receives constant weights
        ``c = logit(p0) / (beta_f * K * a)`` and zero bias, so the lower bound
        on the group logits keeps the initial probability near ``p0``.
        """
        if self.scale_mode != "hybrid":
            print("[init_identity_bias_zero] Warning: this initializer assumes scale_mode == 'hybrid'.")
        a = max(min(self.hybrid_floor_a, 1.0), 1e-6)
        L = self._logit(p0) / float(self.beta_f)
        for group in self.group_defs:
            gname = self._group_name(group)
            K = len(group)
            c = L / (K * a)
            self._weights[gname].fill_(c)
            self._biases[gname].zero_()

    def build_dataloader_cached(self, dataloader):
        """Precompute tree metadata and attributes for a DataLoader.

        The returned DataLoader wraps the original dataset and emits
        ``((x, idx), y)`` batches, where ``idx`` is the original dataset index.
        The layer uses those indexes to reuse cached preprocessing during
        training.
        """
        from torch.utils.data import DataLoader

        dataset_wrapped = IndexedDatasetWrapper(dataloader.dataset)
        new_loader = DataLoader(
            dataset_wrapped,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=False,
            collate_fn=dataloader.collate_fn,
            persistent_workers=getattr(dataloader, "persistent_workers", False),
        )

        print(f"[ConnectedFilterPreprocessingLayer] Preprocessing dataset using mode '{self.scale_mode}'...")
        self._stats_frozen = False
        total_batches = len(new_loader)

        with torch.no_grad():
            for batch_i, ((x, idx), y) in enumerate(new_loader):
                B, C, H, W = x.shape
                for b in range(B):
                    for c in range(C):
                        key = f"{int(idx[b])}_{c}"
                        self._ensure_tree_info_and_attributes_cached(key, x[b, c])
                if (batch_i + 1) % 10 == 0 or batch_i == total_batches - 1:
                    print(f"  [{batch_i+1}/{total_batches}] batches processed.")

        self.freeze_ds_stats()
        self.refresh_cached_normalization()
        print(f"[ConnectedFilterPreprocessingLayer] Full and normalized cache with '{self.scale_mode}'.")
        return new_loader


__all__ = ["ConnectedFilterPreprocessingLayerLegacy"]
