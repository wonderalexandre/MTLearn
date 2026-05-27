"""Reference CFP implementation backed by CPU tree traversal.

This module keeps a CPU-oriented baseline for connected-filter preprocessing.
It delegates the connected filtering and parameter-gradient computation to the
C++ ``ConnectedFilterPreprocessingTreeTraversal`` helper instead of using the implicit
or dense-Jacobian PyTorch formulations. The implementation is useful for
comparison with historical experiments and backend behavior.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from .. import morphology
import mtlearn
from ._helpers import (
    group_name,
    to_numpy_u8,
    build_tree,
    update_ds_stats,
    normalize_with_ds_stats,
    maybe_refresh_norm_for_key,
    make_stats_payload,
    load_stats_payload,
    IndexedDatasetWrapper,
    normalize_attributes_spec,
    validate_attributes_for_tree_type,
)






class ConnectedFilterPreprocessingCPUTreeTraversalFunction(torch.autograd.Function):
    """Autograd function backed by C++ filtering and C++ tree traversal grads."""

    @staticmethod
    def forward(ctx, tree, attrs2d: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, beta_f: float = 1000.0, clamp_logits: bool = True):
        """Apply the connected filter for one tree and one attribute group.

        Args:
            tree: Morphology-tree handle from the C++ backend.
            attrs2d: Normalized attributes with shape ``(num_nodes, K)``.
            weight: Learnable group weight vector with shape ``(K,)``.
            bias: Learnable scalar bias, either ``()`` or ``(1,)``.
            beta_f: Forward sigmoid gain.
            clamp_logits: Whether to clamp ``beta_f * logits`` before sigmoid.

        Returns:
            Filtered image with shape ``(H, W)``.
        """
        assert attrs2d.dim() == 2, "attrs2d must have shape (num_nodes, K)"
        assert weight.dim() == 1, "weight must have shape (K,)"
        logits = attrs2d @ weight.view(-1) + bias.view(())   # (numNodes,)
        s = beta_f * logits
        if clamp_logits:
            s = torch.clamp(s, -12.0, 12.0)
        sigmoid = torch.sigmoid(s)  # (numNodes,)
        y_pred = mtlearn.ConnectedFilterPreprocessingTreeTraversal.filtering(tree, sigmoid)

        # Save the C++ tree handle and tensors needed by the C++ gradient routine.
        ctx.tree = tree
        ctx.beta_f = beta_f
        ctx.save_for_backward(attrs2d, sigmoid)
        return y_pred

    @staticmethod
    def backward(ctx, grad_output):
        """Return gradients for ``weight`` and ``bias`` using C++ traversal."""
        attrs2d, sigmoid = ctx.saved_tensors
        tree = ctx.tree
        beta_f = ctx.beta_f

        # C++: gradients(treePtr, attrs, sigmoid, dL/dY) -> (dW, dB).
        dW, dB = mtlearn.ConnectedFilterPreprocessingTreeTraversal.gradients(
            tree, attrs2d, sigmoid, grad_output
        )

        # Match forward args: (tree, attrs2d, weight, bias, beta_f, clamp_logits)
        return None, None, dW, dB, None, None


class ConnectedFilterPreprocessingLayerWithCPUTreeTraversal(torch.nn.Module):
    """Reference CFP layer that uses C++ CPU tree traversal.

    For each attribute group ``g`` with ``K`` normalized attributes
    ``A_g in R[num_nodes, K]``, the layer computes
    ``sigmoid(beta_f * (A_g @ w_g + b_g))`` and sends that criterion to the C++
    connected-filter backend. The backward pass asks the backend to traverse
    the tree and compute ``dW`` and ``dB``.

    This implementation is CPU-bound and mainly serves as a reference for the
    primary implicit-Jacobian layer.

    Args:
        in_channels: Number of input channels.
        attributes_spec: Attribute groups. Each group must contain at least one
            morphology attribute enum.
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
                 clamp_logits: bool = True,
                 hybrid_k: float = 3.0,
                 hybrid_floor_a: float = 0.05,
                 tos_interpolation=None,
                 tos_infinity_seed_row: int = 0,
                 tos_infinity_seed_col: int = 0,
                 ):
        """Initialize CPU-tree-traversal CFP caches and parameters.

        The constructor keeps the same public configuration as the primary CFP
        layer, but cache entries store backend tree handles because both
        filtering and gradient computation are delegated to C++ tree traversal.
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

        # Tree, attribute, and normalization cache state.
        self._trees      = {}
        self._base_attrs = {}
        self._norm_attrs = {}
        self._stats_epoch = 0
        self._norm_epoch_by_key = {}
        self._ds_stats = {}
        self._stats_frozen = False

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

    def _to_numpy_u8(self, img2d_t: torch.Tensor) -> np.ndarray:
        """Convert one image channel to the backend's ``np.uint8`` format."""
        return to_numpy_u8(img2d_t)

    def _build_tree(self, img_np: np.ndarray):
        """Build the configured morphology tree for one ``np.uint8`` image."""
        return build_tree(
            img_np,
            self.tree_type,
            tos_interpolation=self.tos_interpolation,
            tos_infinity_seed_row=self.tos_infinity_seed_row,
            tos_infinity_seed_col=self.tos_infinity_seed_col,
        )

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

        Hybrid mode applies z-score normalization, clipping, and remapping to
        ``[hybrid_floor_a, 1]``.
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
    def _ensure_tree_and_attr(self, key: str, img_t: torch.Tensor):
        """Ensure tree and raw/normalized attributes exist in cache."""
        if key in self._trees:
            return
        img_np = self._to_numpy_u8(img_t.detach())
        tree = self._build_tree(img_np)
        self._trees[key] = tree
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
    def inspect_training_sample(self, img, channel: int = 0, idx: int | None = None, build_if_missing: bool = True):
        """Return an inspection package for one image.

        The result contains the tree, raw and normalized attributes grouped by
        attribute group name, and references to the current group weights and
        biases.

        Args:
            img: Tensor ``(H, W)`` or ``(C, H, W)``, or ``(img, idx)`` tuple.
            channel: Channel index used when ``img`` has more than one channel.
            idx: Optional sample index, also accepted through ``(img, idx)``.
            build_if_missing: Build and cache preprocessing if it is missing.

        Raises:
            KeyError: If ``build_if_missing`` is false and the key is absent.

        Weights and biases are returned as parameter references; clone them if
        immutable snapshots are needed.
        """
        # Detect whether img was passed as (img, idx).
        if isinstance(img, tuple) and len(img) == 2 and isinstance(img[0], torch.Tensor) and (
            isinstance(img[1], int) or (isinstance(img[1], torch.Tensor) and img[1].numel() == 1)
        ):
            img_tensor = img[0]
            idx_val = int(img[1]) if not isinstance(img[1], torch.Tensor) else int(img[1].item())
        else:
            img_tensor = img
            idx_val = idx

        # Normalize image layout to (C, H, W).
        if img_tensor.dim() == 2:
            imgCHW = img_tensor.unsqueeze(0)  # (1,H,W)
        elif img_tensor.dim() == 3:
            imgCHW = img_tensor               # (C,H,W)
        else:
            raise ValueError(f"img must be (H, W) or (C, H, W); got {tuple(img_tensor.shape)}")

        C, H, W = imgCHW.shape
        if C != self.in_channels:
            # A single-channel input is valid even if the layer has more channels.
            if C != 1:
                raise AssertionError(f"in_channels={self.in_channels}, input C={C}")

        c = channel if C > 1 else 0
        t_u8 = imgCHW[c]

        # Use persistent cache when an index is available.
        if idx_val is not None:
            key = f"{idx_val}_{c}"
            if build_if_missing:
                print(f"[inspect_training_sample] Using cache key '{key}'.")
                self._ensure_tree_and_attr(key, t_u8)
            else:
                print(f"[inspect_training_sample] Using cache key '{key}' without build_if_missing.")
            tree = self._trees[key]
            # Refresh normalized attributes if dataset statistics changed.
            self._maybe_refresh_norm_for_key(key)
            base_attrs_by_group = {}
            norm_attrs_by_group = {}
            weights_by_group    = {}
            bias_by_group       = {}
            for group in self.group_defs:
                gname = self._group_name(group)
                # Stack one column per attribute in the group.
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
            img_np = self._to_numpy_u8(t_u8.detach())
            tree = self._build_tree(img_np)
            per_attr_raw, per_attr_norm = {}, {}
            for attr_type in self._all_attr_types:
                attr_np  = morphology.compute_attributes(tree, [attr_type])[1]
                a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
                a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                per_attr_raw[attr_type]  = a_raw_1d.unsqueeze(1)
                per_attr_norm[attr_type] = a_norm
            base_attrs_by_group = {}
            norm_attrs_by_group = {}
            weights_by_group    = {}
            bias_by_group       = {}
            for group in self.group_defs:
                gname = self._group_name(group)
                cols_raw  = [per_attr_raw[attr_type].view(-1, 1) for attr_type in group]
                cols_norm = [per_attr_norm[attr_type].view(-1, 1) for attr_type in group]
                A_raw  = torch.cat(cols_raw,  dim=1)
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
        """Apply CFP to a batch and return ``(B, C * groups, H, W)`` output."""
        # Support (x, idx) tuples and [x, idx] lists emitted by DataLoader.
        if isinstance(x, tuple) and len(x) == 2:
            x, idx = x
            use_cache = True
        elif isinstance(x, list) and len(x) == 2 and isinstance(x[1], torch.Tensor) and x[1].dim() == 1:
            x, idx = x[0], x[1]
            use_cache = True
        else:
            # Handle a list of tensors without explicit indexes.
            if isinstance(x, list):
                x = torch.stack(x, dim=0)
            idx = torch.arange(x.size(0), device=x.device)
            use_cache = False

        #print(f"[ConnectedFilterPreprocessingLayerWithCPUTreeTraversal] forward: use_cache={use_cache}")

        assert x.dim() == 4, f"expected (B, C, H, W), got {tuple(x.shape)}"
        B, C, H, W = x.shape
        assert C == self.in_channels, f"in_channels={self.in_channels}, input C={C}"

        out = torch.empty((B, self.out_channels, H, W), dtype=torch.float32, device=self.device)

        for b in range(B):
            for c in range(C):
                if use_cache:
                    key = f"{int(idx[b])}_{c}"
                    self._ensure_tree_and_attr(key, x[b, c])
                    tree = self._trees[key]
                    self._maybe_refresh_norm_for_key(key)
                    for g, group in enumerate(self.group_defs):
                        gname = self._group_name(group)
                        # Build A_norm by stacking one normalized column per group attribute.
                        cols = [self._norm_attrs[key][attr_type].view(-1, 1) for attr_type in group]
                        A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                        y_ch = ConnectedFilterPreprocessingCPUTreeTraversalFunction.apply(
                            tree, A_norm, self._weights[gname], self._biases[gname],
                            self.beta_f, self.clamp_logits
                        )
                        if self.top_hat:
                            x_bc = x[b, c].to(dtype=torch.float32, device=self.device)
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
                    #print(f"[ConnectedFilterPreprocessingLayerWithCPUTreeTraversal] forward: computing tree/attrs directly for sample {b}, channel {c}")
                    img_np = self._to_numpy_u8(x[b, c].detach())
                    tree = self._build_tree(img_np)
                    per_attr_norm = {}
                    for attr_type in self._all_attr_types:
                        attr_np = morphology.compute_attributes(tree, [attr_type])[1]
                        a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
                        a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                        per_attr_norm[attr_type] = a_norm
                    for g, group in enumerate(self.group_defs):
                        gname = self._group_name(group)
                        cols = [per_attr_norm[attr_type].view(-1, 1) for attr_type in group]
                        A_norm = torch.cat(cols, dim=1)
                        y_ch = ConnectedFilterPreprocessingCPUTreeTraversalFunction.apply(
                            tree, A_norm, self._weights[gname], self._biases[gname],
                            self.beta_f, self.clamp_logits
                        )
                        if self.top_hat:
                            x_bc = x[b, c].to(dtype=torch.float32, device=self.device)
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
    def predict(self, x: torch.Tensor, beta_f: float = 1.0) -> torch.Tensor:
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

            #print(f"[ConnectedFilterPreprocessingLayerWithCPUTreeTraversal] predict: use_cache={use_cache}")

            B, C, H, W = x.shape
            out = torch.empty((B, self.out_channels, H, W), dtype=torch.float32, device=self.device)
            for b in range(B):
                for c in range(C):
                    if use_cache:
                        key = f"{int(idx[b])}_{c}"
                        self._ensure_tree_and_attr(key, x[b, c])
                        tree = self._trees[key]
                        self._maybe_refresh_norm_for_key(key)
                        for g, group in enumerate(self.group_defs):
                            gname = self._group_name(group)
                            cols = [self._norm_attrs[key][attr_type].view(-1, 1) for attr_type in group]
                            A_norm = torch.cat(cols, dim=1)  # (numNodes, K)
                            y_ch = ConnectedFilterPreprocessingCPUTreeTraversalFunction.apply(
                                tree, A_norm, self._weights[gname], self._biases[gname],
                                beta_f, self.clamp_logits  # caller-provided beta_f
                            )
                            if self.top_hat:
                                x_bc = x[b, c].to(dtype=torch.float32, device=self.device)
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
                        #print(f"[ConnectedFilterPreprocessingLayerWithCPUTreeTraversal] predict: computing tree/attrs directly for sample {b}, channel {c}")
                        img_np = self._to_numpy_u8(x[b, c].detach())
                        tree = self._build_tree(img_np)
                        per_attr_norm = {}
                        for attr_type in self._all_attr_types:
                            attr_np = morphology.compute_attributes(tree, [attr_type])[1]
                            a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
                            a_norm = self._normalize_with_ds_stats(attr_type, a_raw_1d)
                            per_attr_norm[attr_type] = a_norm
                        for g, group in enumerate(self.group_defs):
                            gname = self._group_name(group)
                            cols = [per_attr_norm[attr_type].view(-1, 1) for attr_type in group]
                            A_norm = torch.cat(cols, dim=1)
                            y_ch = ConnectedFilterPreprocessingCPUTreeTraversalFunction.apply(
                                tree, A_norm, self._weights[gname], self._biases[gname],
                                beta_f, self.clamp_logits
                            )
                            if self.top_hat:
                                x_bc = x[b, c].to(dtype=torch.float32, device=self.device)
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
        ``logit(p0) / beta_f``.
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
        ``c = logit(p0) / (beta_f * K * a)`` and zero bias.
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
        """Precompute trees and attributes for a DataLoader.

        The returned DataLoader wraps the original dataset and emits
        ``((x, idx), y)`` batches. This method builds per-sample trees and
        attributes, updates dataset normalization statistics, refreshes cached
        normalized attributes, and freezes statistics when preprocessing ends.
        """
        from torch.utils.data import Dataset, DataLoader

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

        print(f"[ConnectedFilterPreprocessingLayerWithCPUTreeTraversal] Preprocessing dataset using mode '{self.scale_mode}'...")
        self._stats_frozen = False
        total_batches = len(new_loader)

        with torch.no_grad():
            for batch_i, ((x, idx), y) in enumerate(new_loader):
                B, C, H, W = x.shape
                for b in range(B):
                    for c in range(C):
                        key = f"{int(idx[b])}_{c}"
                        self._ensure_tree_and_attr(key, x[b, c])
                if (batch_i + 1) % 10 == 0 or batch_i == total_batches - 1:
                    print(f"  [{batch_i+1}/{total_batches}] batches processed.")

        self.freeze_ds_stats()
        self.refresh_cached_normalization()
        print(f"[ConnectedFilterPreprocessingLayerWithCPUTreeTraversal] Full and normalized cache with '{self.scale_mode}'.")
        return new_loader

CFPLayerWithCPUTreeTraversal = ConnectedFilterPreprocessingLayerWithCPUTreeTraversal

__all__ = [
    'ConnectedFilterPreprocessingLayerWithCPUTreeTraversal',
    'CFPLayerWithCPUTreeTraversal',
    'ConnectedFilterPreprocessingCPUTreeTraversalFunction',
]
