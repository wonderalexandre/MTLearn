"""Primary connected-filter preprocessing layer.

This module implements the production CFP layer used by mtlearn experiments.
It avoids materializing the dense tree-to-pixel Jacobian during reconstruction
and backward propagation. Instead, it uses preorder/postorder tree metadata to
apply the equivalent operations with linear memory in the number of nodes and
pixels.

Tree construction and attribute computation are performed through
``mtlearn.morphology`` and are intentionally outside the autograd path. The
learnable parameters are the per-attribute-group weight vectors and biases that
produce the node-wise sigmoid filtering criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
import re
from typing import Any, Mapping
import torch
import numpy as np
from .. import morphology
import mtlearn
from ._helpers import (
    to_numpy_u8,
    build_tree,
    update_ds_stats,
    normalize_with_ds_stats,
    IndexedDatasetWrapper,
    normalize_attributes_spec,
    validate_attributes_for_tree_type,
)




class ConnectedFilterPreprocessingImplicitJacobianFunction(torch.autograd.Function):
    """Autograd function for CFP with an implicit morphology-tree Jacobian.

    The forward reconstruction is mathematically equivalent to
    ``J.T @ filtered_residues`` where ``J`` is the dense node-to-pixel
    Jacobian, but the implementation uses tree entry/exit times and a prefix
    scan instead of materializing ``J``.

    Conceptual dense form:

    ```
    enter_n = tpre.view(-1, 1)
    exit_n  = tpost.view(-1, 1)
    enter_p = tpre[node_of_pixel].view(1, -1)
    J = (enter_n <= enter_p) & (enter_p < exit_n)
    return J.float().T @ filtered_res
    ```
    """
    @staticmethod
    def forward_from_info(filtered_res, tpre, tpost, node_of_pixel, parent, order_forward=None):
        """Reconstruct pixels from filtered residues without a dense Jacobian.

        ``filtered_res`` stores a value per tree node. The prefix-scan over
        ``tpre``/``tpost`` accumulates all active ancestor residues for each
        pixel's canonical node.
        """

        max_t = int(tpost.max().item()) + 1
        delta = torch.zeros(max_t, device=filtered_res.device, dtype=filtered_res.dtype)
        delta.index_add_(0, tpre, filtered_res)
        delta.index_add_(0, tpost, -filtered_res)
        y_cumsum = torch.cumsum(delta, dim=0)
        y = y_cumsum[tpre[node_of_pixel]]
        return y

    def backward_from_info(grad_output, tpre, tpost, parent, node_of_pixel, order_pre=None):
        """Propagate pixel gradients back to tree nodes without a dense matrix.

        This computes the equivalent of multiplying by the dense Jacobian
        ``J`` used by the explicit implementation. Pixel gradients are first
        accumulated on their canonical nodes and then prefix sums over preorder
        intervals recover the total gradient for each tree node.
        """
        g_pix = grad_output.reshape(-1)
        N = tpre.numel()
        base = torch.zeros(N, dtype=g_pix.dtype, device=g_pix.device)
        base.index_add_(0, node_of_pixel.reshape(-1), g_pix)

        # Preorder permutation and inverse rank.
        if( order_pre is None):
            order_pre = torch.argsort(tpre)
        pre_rank = torch.empty_like(order_pre)
        pre_rank[order_pre] = torch.arange(N, device=order_pre.device)

        base_sorted = base[order_pre]
        pref = torch.cumsum(base_sorted, dim=0)
        pref0 = torch.cat([pref.new_zeros(1), pref], dim=0)

        # time -> rank mapping: R[t] = number of nodes with tpre < t.
        T = int(torch.max(tpost).item()) + 1
        counts = torch.bincount(tpre, minlength=T)
        cum = torch.cumsum(counts, dim=0)
        R = torch.cat([cum.new_zeros(1), cum[:-1]], dim=0)

        l = pre_rank
        r = R[tpost]                                         # exclusive end
        grad_nodes = pref0[r] - pref0[l]
        return grad_nodes

    @staticmethod
    def forward(
        ctx,
        weight,
        bias,
        residues,
        tpre,
        tpost,
        parent,
        node_of_pixel,
        attrs2d,
        numRows: int,
        numCols: int,
        beta_f: float = 1.0,
        clamp_min=None,
        clamp_max=None,
        order_forward=None,
        order_backward=None,
        preserve_root: bool = False,
    ):
        """Apply the connected filter using implicit reconstruction metadata.

        Args:
            weight, bias: learnable group parameters.
            residues: tree residues, one value per node.
            tpre: node entry times in preorder.
            tpost: node exit times.
            parent: parent node index for every node.
            node_of_pixel: mapping from flattened pixels to tree nodes.
            attrs2d: normalized attributes with shape ``(num_nodes, K)``.
            numRows, numCols: output image dimensions.
            beta_f: sigmoid gain used in the forward pass.
            clamp_min, clamp_max: optional clamp bounds for ``beta_f * logits``
                before sigmoid.
            order_forward: optional cached order for forward reconstruction.
            order_backward: optional cached order for gradient propagation.
            preserve_root: Whether to force the root-node gate to one.

        Returns:
            Filtered image with shape ``(numRows, numCols)``.
        """
        # Node-wise sigmoid criterion.
        logits = attrs2d @ weight.view(-1) + bias
        s = beta_f * logits
        if isinstance(clamp_min, bool) and clamp_max is None:
            clamp_min, clamp_max = (-12.0, 12.0) if clamp_min else (None, None)
        if (clamp_min is None) != (clamp_max is None):
            raise ValueError("clamp_min and clamp_max must be provided together.")
        if clamp_min is not None and clamp_max is not None:
            if clamp_min >= clamp_max:
                raise ValueError("clamp_min must be smaller than clamp_max.")
            clamp_mask = (s >= clamp_min) & (s <= clamp_max)
            s = torch.clamp(s, clamp_min, clamp_max)
        else:
            clamp_mask = torch.ones_like(s, dtype=torch.bool)
        sigmoid = torch.sigmoid(s)
        if preserve_root:
            node_ids = torch.arange(parent.numel(), device=parent.device)
            root_mask = (parent == node_ids) & (tpost > tpre)
            sigmoid = torch.where(root_mask, torch.ones_like(sigmoid), sigmoid)

        # Implicit reconstruction from filtered node residues to pixels.
        filtered_res = residues * sigmoid
        y = ConnectedFilterPreprocessingImplicitJacobianFunction.forward_from_info(filtered_res, tpre, tpost, node_of_pixel, parent, order_forward)
        y_2d = y.reshape(numRows, numCols)

        # Backward context: only tensors needed to compute dW and dB are saved.
        ctx.save_for_backward(attrs2d, residues, sigmoid, clamp_mask, tpre, tpost, parent, node_of_pixel)
        ctx.beta_f = beta_f
        ctx.order_backward = order_backward
        return y_2d

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients for the learnable criterion parameters.

        Gradients flow to ``weight`` and ``bias``. Tree topology, attributes,
        residues, and image dimensions are treated as fixed preprocessing data.
        """
        # Recover the tensors needed by the implicit Jacobian computation.
        attrs2d, residues, sigmoid, clamp_mask, tpre, tpost, parent, node_of_pixel = ctx.saved_tensors
        beta_f = ctx.beta_f
        order_backward = ctx.order_backward
        grad_output_flat = grad_output.flatten()

        # Implicit tree backward equivalent to J @ grad_output.
        grad_nodes = ConnectedFilterPreprocessingImplicitJacobianFunction.backward_from_info(
            grad_output_flat, tpre, tpost, parent, node_of_pixel, order_backward
        )

        # Chain rule through the sigmoid criterion.
        d_sigmoid = sigmoid * (1 - sigmoid)
        grad_s = grad_nodes * residues * d_sigmoid * beta_f
        grad_s = torch.where(clamp_mask, grad_s, torch.zeros_like(grad_s))

        # Final gradients for the group weight vector and scalar bias.
        dW = attrs2d.T @ grad_s
        dB = grad_s.sum().view(1)

        # Return one gradient slot for each forward argument.
        return (
            dW,          # weight
            dB,          # bias
            None,        # residues
            None,        # tpre
            None,        # tpost
            None,        # parent
            None,        # node_of_pixel
            None,        # attrs2d
            None,        # numRows
            None,        # numCols
            None,        # beta_f
            None,        # clamp_min
            None,        # clamp_max
            None,        # order_forward
            None,        # order_backward
            None         # preserve_root
        )




@dataclass(frozen=True)
class CFPValuation:
    """Signal reconstructed by one CFP filter.

    ``ALTITUDE`` reconstructs the filtered image altitude, ``ALTITUDE_TOPHAT``
    reconstructs the tree-type-specific altitude top-hat, and
    ``node_attribute`` reconstructs a scalar node attribute.
    """

    kind: str
    attribute: Any = None

    @classmethod
    def node_attribute(cls, attribute: Any) -> "CFPValuation":
        """Use a scalar node attribute as the valuation to be filtered."""
        return cls("node_attribute", attribute)


CFPValuation.ALTITUDE = CFPValuation("altitude")
CFPValuation.ALTITUDE_TOPHAT = CFPValuation("altitude_tophat")


@dataclass(frozen=True)
class _NormalizedFilterSpec:
    index: int
    key: str
    tree_type: str
    tree_key: str
    attributes: tuple[Any, ...]
    valuation: CFPValuation
    valuation_key: str
    preserve_root: bool
    monotonicity_weight: float
    tos_interpolation: Any
    tos_infinity_seed_row: int
    tos_infinity_seed_col: int


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def _is_altitude_attribute(value: Any) -> bool:
    return _enum_name(value) in {"ALTITUDE", "LEVEL"}


def _normalize_valuation(value: Any) -> CFPValuation:
    if value is None:
        return CFPValuation.ALTITUDE
    if isinstance(value, CFPValuation):
        if value.kind == "node_attribute" and _is_altitude_attribute(value.attribute):
            return CFPValuation.ALTITUDE
        return value
    if _is_altitude_attribute(value):
        return CFPValuation.ALTITUDE
    return CFPValuation.node_attribute(value)


def _valuation_key(valuation: CFPValuation) -> str:
    if valuation.kind == "node_attribute":
        return f"node_attribute:{_enum_name(valuation.attribute)}"
    return valuation.kind


def _uses_altitude_signal(valuation: CFPValuation) -> bool:
    return valuation.kind in {"altitude", "altitude_tophat"}


def _is_altitude_tophat_valuation(valuation: CFPValuation) -> bool:
    return valuation.kind == "altitude_tophat"


def _normalize_clamp(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("clamp must be None, a positive scalar, or a (min, max) pair.")
    if isinstance(value, numbers.Real):
        bound = float(value)
        if not math.isfinite(bound) or bound <= 0.0:
            raise ValueError("scalar clamp must be finite and positive.")
        return (-bound, bound)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        clamp_min = float(value[0])
        clamp_max = float(value[1])
        if not math.isfinite(clamp_min) or not math.isfinite(clamp_max):
            raise ValueError("clamp bounds must be finite.")
        if clamp_min >= clamp_max:
            raise ValueError("clamp bounds must satisfy min < max.")
        return (clamp_min, clamp_max)
    raise TypeError("clamp must be None, a positive scalar, or a (min, max) pair.")


def _normalize_nonnegative_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a non-negative finite scalar.")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a non-negative finite scalar.")
    return scalar


def _normalize_attribute_dtype(value: Any) -> np.dtype:
    if value is None:
        return np.dtype(np.float32)
    if isinstance(value, torch.dtype):
        if value == torch.float32:
            return np.dtype(np.float32)
        if value == torch.float64:
            return np.dtype(np.float64)
        raise ValueError("attribute_dtype must be np.float32, np.float64, torch.float32, or torch.float64.")
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise TypeError("attribute_dtype must be np.float32, np.float64, torch.float32, or torch.float64.") from exc
    if dtype == np.dtype(np.float32) or dtype == np.dtype(np.float64):
        return dtype
    raise ValueError("attribute_dtype must be np.float32, np.float64, torch.float32, or torch.float64.")


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


class ConnectedFilterPreprocessingLayer(torch.nn.Module):
    """Learnable CFP layer defined by per-output filter specifications.

    Each item in ``filter_specs`` defines one output per input channel:
    morphology tree, scoring attributes, and reconstructed valuation. The
    default valuation is ``CFPValuation.ALTITUDE``. Top-hat output is selected
    with ``CFPValuation.ALTITUDE_TOPHAT``. ``clamp`` optionally bounds
    ``beta_f * logits`` before the sigmoid.

    Tree construction and attribute computation happen outside autograd. The
    trainable parameters are the per-filter weight vectors and scalar biases
    that produce node-wise sigmoid gates from normalized attributes.
    """

    def __init__(
        self,
        in_channels,
        filter_specs,
        *,
        device="cpu",
        scale_mode: str = "hybrid",
        eps: float = 1e-6,
        beta_f: float = 1.0,
        clamp=None,
        hybrid_k: float = 3.0,
        hybrid_floor_a: float = 0.05,
        attribute_dtype=None,
        tos_interpolation=None,
        tos_infinity_seed_row: int = 0,
        tos_infinity_seed_col: int = 0,
    ):
        """Initialize CFP filters, caches, and learnable spec parameters.

        Args:
            in_channels: Number of input image channels expected by ``forward``.
            filter_specs: Iterable of mappings. Each mapping must define
                ``tree_type`` and ``attributes`` and may define ``name``,
                ``valuation``, ``preserve_root``, ``monotonicity_weight``,
                ``tos_interpolation``, ``tos_infinity_seed_row``, and
                ``tos_infinity_seed_col``.
                One output channel is produced for each input channel and each
                filter spec.
            device: Device used for CFP tensors and trainable parameters.
                Morphology-tree construction itself runs in the native CPU
                backend.
            scale_mode: Attribute normalization mode. ``"hybrid"`` uses
                dataset-level z-score statistics followed by clipping/rescaling;
                ``"minmax01"``, ``"zscore_tree"``, and ``"none"`` are also
                supported by the shared normalization helpers.
            eps: Numerical floor used by normalization.
            beta_f: Sigmoid gain used during ``forward``.
            clamp: Optional bound applied to ``beta_f * logits`` before the
                sigmoid. Use ``None`` for no clamp, a positive scalar for
                symmetric bounds, or ``(min, max)`` for explicit bounds.
            hybrid_k: Clipping radius used by ``scale_mode="hybrid"``.
            hybrid_floor_a: Lower endpoint used by hybrid rescaling.
            attribute_dtype: Floating dtype used for morphology attribute
                extraction, cache storage, and normalization. Accepts
                ``np.float32``, ``np.float64``, ``torch.float32``,
                ``torch.float64``, and equivalent NumPy dtype strings. ``None``
                keeps the historical ``np.float32`` default.
            tos_interpolation: Default tree-of-shapes interpolation for specs
                that do not override it.
            tos_infinity_seed_row: Default tree-of-shapes infinity seed row.
            tos_infinity_seed_col: Default tree-of-shapes infinity seed column.

        Raises:
            ValueError: If ``filter_specs`` is empty or a spec is invalid.
            TypeError: If a spec or clamp value has an unsupported type.
        """
        super().__init__()

        self.hybrid_k = float(hybrid_k)
        self.hybrid_floor_a = float(hybrid_floor_a)
        self.in_channels = int(in_channels)
        self.device = torch.device(device)
        self.scale_mode = str(scale_mode)
        self.eps = float(eps)
        self.beta_f = float(beta_f)
        self.clamp = _normalize_clamp(clamp)
        self.attribute_dtype = _normalize_attribute_dtype(attribute_dtype)

        self.filter_specs = self._normalize_filter_specs(
            filter_specs,
            default_tos_interpolation=tos_interpolation,
            default_tos_infinity_seed_row=int(tos_infinity_seed_row),
            default_tos_infinity_seed_col=int(tos_infinity_seed_col),
        )
        self.num_specs = len(self.filter_specs)
        self.out_channels = self.in_channels * self.num_specs

        self._spec_by_key = {spec.key: spec for spec in self.filter_specs}
        self._tree_spec_by_key = {}
        self._scoring_attrs_by_tree_key = {}
        self._valuations_by_tree_key = {}
        for spec in self.filter_specs:
            self._tree_spec_by_key.setdefault(spec.tree_key, spec)
            self._scoring_attrs_by_tree_key.setdefault(spec.tree_key, set()).update(spec.attributes)
            self._valuations_by_tree_key.setdefault(spec.tree_key, set()).add(spec.valuation)

        self._tree_info = {}
        self._base_attrs = {}
        self._norm_attrs = {}
        self._valuation_increments = {}
        self._stats_epoch = 0
        self._norm_epoch_by_key = {}
        self._ds_stats = {}
        self._stats_frozen = False

        self._weights = torch.nn.ParameterDict()
        self._biases = torch.nn.ParameterDict()
        for spec in self.filter_specs:
            k = len(spec.attributes)
            w = torch.empty(k, dtype=torch.float32, device=self.device)
            b = torch.empty(1, dtype=torch.float32, device=self.device)
            fan_in, fan_out = k, 1
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            torch.nn.init.uniform_(w, -math.sqrt(3.0) * std, math.sqrt(3.0) * std)
            torch.nn.init.constant_(b, 0.0)
            self._weights[spec.key] = torch.nn.Parameter(w, requires_grad=True)
            self._biases[spec.key] = torch.nn.Parameter(b, requires_grad=True)

    @staticmethod
    def _tree_key(tree_type, tos_interpolation, tos_infinity_seed_row, tos_infinity_seed_col) -> str:
        interpolation_name = _enum_name(tos_interpolation) if tos_interpolation is not None else "None"
        return f"{tree_type}|{interpolation_name}|{tos_infinity_seed_row}|{tos_infinity_seed_col}"

    def _normalize_filter_specs(
        self,
        filter_specs,
        *,
        default_tos_interpolation,
        default_tos_infinity_seed_row: int,
        default_tos_infinity_seed_col: int,
    ):
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
                raise ValueError("output_mode was removed; use CFPValuation.ALTITUDE_TOPHAT for top-hat output.")

            spec_name = _normalize_filter_spec_name(raw_spec.get("name", None), index, seen_names)
            tree_type = morphology.normalize_tree_type(raw_spec["tree_type"])
            raw_attributes = raw_spec["attributes"]
            raw_group = tuple(raw_attributes) if isinstance(raw_attributes, (list, tuple)) else (raw_attributes,)
            if len(raw_group) < 1:
                raise ValueError("Each filter spec must contain at least one attribute.")

            attributes = normalize_attributes_spec([raw_group], tree_type)[0][0]
            validate_attributes_for_tree_type(attributes, tree_type)

            valuation = _normalize_valuation(raw_spec.get("valuation", None))
            self._validate_valuation_for_tree_type(valuation, tree_type)

            spec_tos_interpolation = raw_spec.get("tos_interpolation", default_tos_interpolation)
            if tree_type == morphology.TreeType.TREE_OF_SHAPES.value:
                spec_tos_interpolation = morphology.normalize_tos_interpolation(spec_tos_interpolation)
            spec_tos_infinity_seed_row = int(raw_spec.get("tos_infinity_seed_row", default_tos_infinity_seed_row))
            spec_tos_infinity_seed_col = int(raw_spec.get("tos_infinity_seed_col", default_tos_infinity_seed_col))
            tree_key = self._tree_key(
                tree_type,
                spec_tos_interpolation,
                spec_tos_infinity_seed_row,
                spec_tos_infinity_seed_col,
            )

            normalized.append(
                _NormalizedFilterSpec(
                    index=index,
                    key=spec_name,
                    tree_type=tree_type,
                    tree_key=tree_key,
                    attributes=tuple(attributes),
                    valuation=valuation,
                    valuation_key=_valuation_key(valuation),
                    preserve_root=bool(raw_spec.get("preserve_root", False)),
                    monotonicity_weight=_normalize_nonnegative_scalar(
                        raw_spec.get("monotonicity_weight", 0.0),
                        "monotonicity_weight",
                    ),
                    tos_interpolation=spec_tos_interpolation,
                    tos_infinity_seed_row=spec_tos_infinity_seed_row,
                    tos_infinity_seed_col=spec_tos_infinity_seed_col,
                )
            )

        if not normalized:
            raise ValueError("filter_specs must contain at least one filter specification.")
        return tuple(normalized)

    @staticmethod
    def _validate_valuation_for_tree_type(valuation: CFPValuation, tree_type: str) -> None:
        if valuation.kind in {"altitude", "altitude_tophat"}:
            return
        if valuation.kind != "node_attribute":
            raise ValueError(f"unknown CFP valuation kind: {valuation.kind!r}")
        if isinstance(valuation.attribute, morphology.AttributeGroup):
            raise ValueError("CFPValuation.node_attribute expects one scalar attribute, not an AttributeGroup.")
        validate_attributes_for_tree_type([valuation.attribute], tree_type)

    def _stat_key(self, tree_key: str, attr_type: Any) -> str:
        return f"{tree_key}::{_enum_name(attr_type)}"

    def _to_numpy_u8(self, img2d_t: torch.Tensor) -> np.ndarray:
        return to_numpy_u8(img2d_t)

    def _build_tree(self, img_np: np.ndarray, spec: _NormalizedFilterSpec):
        return build_tree(
            img_np,
            spec.tree_type,
            tos_interpolation=spec.tos_interpolation,
            tos_infinity_seed_row=spec.tos_infinity_seed_row,
            tos_infinity_seed_col=spec.tos_infinity_seed_col,
        )

    def _compute_tree_info(self, tree, spec: _NormalizedFilterSpec):
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
            "tree_type": spec.tree_type,
            "order_forward": torch.argsort(tpre, descending=False).to(self.device),
            "order_backward": torch.argsort(tpre).to(self.device),
        }
        return info

    def _update_ds_stats(self, stat_key, a_raw_1d: torch.Tensor):
        if getattr(self, "_stats_frozen", False):
            return
        smode = "zscore_tree" if self.scale_mode == "hybrid" else self.scale_mode
        changed = update_ds_stats(self._ds_stats, smode, stat_key, a_raw_1d)
        if changed:
            self._stats_epoch += 1

    def _normalize_with_ds_stats(self, stat_key, a_raw_1d: torch.Tensor) -> torch.Tensor:
        if self.scale_mode != "hybrid":
            return normalize_with_ds_stats(self._ds_stats, self.scale_mode, self.eps, stat_key, a_raw_1d)

        st = self._ds_stats.get(stat_key, None)
        if st is None:
            raise RuntimeError(
                "scale_mode='hybrid' requires dataset statistics. "
                "Call build_dataloader_cached(...) or load_stats(...) before forward/inspection."
            )
        count = st["count"].to(dtype=st["sum"].dtype, device=st["sum"].device)
        mean = st["sum"] / torch.clamp(count, min=1.0)
        var = st["sumsq"] / torch.clamp(count, min=1.0) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=self.eps))
        x = (a_raw_1d - mean) / std
        k = torch.tensor(self.hybrid_k, dtype=x.dtype, device=x.device)
        x = torch.clamp(x, -k, k)
        a = torch.tensor(self.hybrid_floor_a, dtype=x.dtype, device=x.device)
        return a + (1.0 - a) * ((x + k) / (2.0 * k))

    def _compute_valuation_increment(self, tree, info, valuation: CFPValuation) -> torch.Tensor:
        if _uses_altitude_signal(valuation):
            return info["residues"]

        attr_type = valuation.attribute
        attr_np = morphology.compute_attributes(tree, [attr_type], dtype=self.attribute_dtype)[1]
        values = torch.as_tensor(attr_np, device=self.device).squeeze(1)
        parent = info["parent"]
        parent_values = values[parent.clamp_min(0)]
        increments = values - parent_values
        root_or_self = parent == torch.arange(parent.numel(), device=parent.device)
        increments = torch.where(root_or_self, values, increments)
        alive = info["tpost"] > info["tpre"]
        return torch.where(alive, increments, torch.zeros_like(increments))

    def _compute_tree_payload(self, img_np: np.ndarray, tree_key: str, *, update_stats: bool):
        spec = self._tree_spec_by_key[tree_key]
        tree = self._build_tree(img_np, spec)
        info = self._compute_tree_info(tree, spec)

        base_attrs = {}
        norm_attrs = {}
        for attr_type in self._scoring_attrs_by_tree_key.get(tree_key, ()):
            attr_np = morphology.compute_attributes(tree, [attr_type], dtype=self.attribute_dtype)[1]
            a_raw_1d = torch.as_tensor(attr_np, device=self.device).squeeze(1)
            stat_key = self._stat_key(tree_key, attr_type)
            if update_stats:
                self._update_ds_stats(stat_key, a_raw_1d)
            a_norm = self._normalize_with_ds_stats(stat_key, a_raw_1d)
            base_attrs[attr_type] = a_raw_1d.unsqueeze(1)
            norm_attrs[attr_type] = a_norm

        valuation_increments = {}
        for valuation in self._valuations_by_tree_key.get(tree_key, ()):
            valuation_increments[_valuation_key(valuation)] = self._compute_valuation_increment(tree, info, valuation)

        return {
            "info": info,
            "base_attrs": base_attrs,
            "norm_attrs": norm_attrs,
            "valuation_increments": valuation_increments,
        }

    def _ensure_tree_payload_cached(
        self,
        base_key: str,
        img_t: torch.Tensor,
        tree_key: str,
        *,
        update_stats: bool = True,
    ) -> None:
        if base_key in self._tree_info and tree_key in self._tree_info[base_key]:
            return

        img_np = self._to_numpy_u8(img_t.detach())
        payload = self._compute_tree_payload(img_np, tree_key, update_stats=update_stats)
        self._tree_info.setdefault(base_key, {})[tree_key] = payload["info"]
        self._base_attrs.setdefault(base_key, {})[tree_key] = payload["base_attrs"]
        self._norm_attrs.setdefault(base_key, {})[tree_key] = payload["norm_attrs"]
        self._valuation_increments.setdefault(base_key, {})[tree_key] = payload["valuation_increments"]
        self._norm_epoch_by_key[base_key] = self._stats_epoch

    def _require_fixed_dataset_stats(self) -> None:
        if self.scale_mode == "none":
            return

        missing = []
        for tree_key, attr_types in self._scoring_attrs_by_tree_key.items():
            for attr_type in attr_types:
                stat_key = self._stat_key(tree_key, attr_type)
                if stat_key not in self._ds_stats:
                    missing.append(stat_key)

        if missing:
            shown = ", ".join(missing[:3])
            suffix = "" if len(missing) <= 3 else f", ... ({len(missing)} total)"
            raise RuntimeError(
                "build_dataloader_cached_fixed_stats(...) requires fixed dataset statistics. "
                "Call build_dataloader_cached(...) on the training split or load_stats(...) first. "
                f"Missing stats: {shown}{suffix}"
            )

    def _maybe_refresh_norm_for_key(self, base_key: str) -> None:
        if base_key not in self._base_attrs:
            return
        if self._norm_epoch_by_key.get(base_key, -1) == self._stats_epoch:
            return

        refreshed = {}
        for tree_key, per_attr_raw in self._base_attrs[base_key].items():
            refreshed[tree_key] = {}
            for attr_type, a_raw_2d in per_attr_raw.items():
                stat_key = self._stat_key(tree_key, attr_type)
                refreshed[tree_key][attr_type] = self._normalize_with_ds_stats(stat_key, a_raw_2d.view(-1))
        self._norm_attrs[base_key] = refreshed
        self._norm_epoch_by_key[base_key] = self._stats_epoch

    def _normalized_attribute_matrix(self, spec: _NormalizedFilterSpec, norm_attrs, *, dtype: torch.dtype):
        cols = [norm_attrs[attr_type].view(-1, 1).to(dtype=dtype, device=self.device) for attr_type in spec.attributes]
        return torch.cat(cols, dim=1)

    def _gate_scores(self, spec: _NormalizedFilterSpec, info, norm_attrs) -> torch.Tensor:
        weight = self._weights[spec.key]
        bias = self._biases[spec.key]
        attrs2d = self._normalized_attribute_matrix(spec, norm_attrs, dtype=weight.dtype)
        logits = attrs2d @ weight.view(-1) + bias
        s = self.beta_f * logits
        if self.clamp is not None:
            s = torch.clamp(s, self.clamp[0], self.clamp[1])
        scores = torch.sigmoid(s)
        if spec.preserve_root:
            parent = info["parent"]
            node_ids = torch.arange(parent.numel(), device=parent.device)
            root_mask = (parent == node_ids) & (info["tpost"] > info["tpre"])
            scores = torch.where(root_mask, torch.ones_like(scores), scores)
        return scores

    def _monotonicity_penalty_for_spec(self, spec: _NormalizedFilterSpec, info, norm_attrs) -> torch.Tensor:
        scores = self._gate_scores(spec, info, norm_attrs)
        parent = info["parent"]
        node_ids = torch.arange(parent.numel(), device=parent.device)
        alive = info["tpost"] > info["tpre"]
        parent_alive = alive[parent]
        edge_mask = alive & parent_alive & (parent != node_ids)
        if not bool(edge_mask.any().item()):
            return scores.sum() * 0.0

        violations = torch.relu(scores[edge_mask] - scores[parent[edge_mask]])
        return float(spec.monotonicity_weight) * violations.square().mean()

    def _zero_parameter_penalty(self) -> torch.Tensor:
        parameter = next(self.parameters())
        return parameter.sum() * 0.0

    def _get_tree_payload_for_sample(self, base_key: str, img_ch: torch.Tensor, spec: _NormalizedFilterSpec, direct_payloads, *, use_cache: bool):
        if use_cache:
            self._ensure_tree_payload_cached(base_key, img_ch, spec.tree_key)
            self._maybe_refresh_norm_for_key(base_key)
            return (
                self._tree_info[base_key][spec.tree_key],
                self._norm_attrs[base_key][spec.tree_key],
                self._valuation_increments[base_key][spec.tree_key],
            )

        if spec.tree_key not in direct_payloads:
            img_np = self._to_numpy_u8(img_ch.detach())
            direct_payloads[spec.tree_key] = self._compute_tree_payload(
                img_np,
                spec.tree_key,
                update_stats=False,
            )
        payload = direct_payloads[spec.tree_key]
        return payload["info"], payload["norm_attrs"], payload["valuation_increments"]

    def _apply_spec(self, spec: _NormalizedFilterSpec, info, norm_attrs, valuation_increments, beta_f):
        weight = self._weights[spec.key]
        dtype = weight.dtype
        clamp_min, clamp_max = self.clamp if self.clamp is not None else (None, None)

        A_norm = self._normalized_attribute_matrix(spec, norm_attrs, dtype=dtype)
        increments = valuation_increments[spec.valuation_key].to(dtype=dtype, device=self.device)
        y_ch = ConnectedFilterPreprocessingImplicitJacobianFunction.apply(
            weight,
            self._biases[spec.key],
            increments,
            info["tpre"],
            info["tpost"],
            info["parent"],
            info["node_of_pixel"],
            A_norm,
            info["numRows"],
            info["numCols"],
            beta_f,
            clamp_min,
            clamp_max,
            info["order_forward"],
            info["order_backward"],
            spec.preserve_root,
        )

        if not _is_altitude_tophat_valuation(spec.valuation):
            return y_ch

        base = ConnectedFilterPreprocessingImplicitJacobianFunction.forward_from_info(
            increments,
            info["tpre"],
            info["tpost"],
            info["node_of_pixel"],
            info["parent"],
            info["order_forward"],
        ).reshape(info["numRows"], info["numCols"])
        if spec.tree_type == morphology.TreeType.MAX_TREE.value:
            return base - y_ch
        if spec.tree_type == morphology.TreeType.MIN_TREE.value:
            return y_ch - base
        return torch.abs(y_ch - base)

    def _batch_input(self, x):
        if isinstance(x, tuple) and len(x) == 2:
            return x[0], x[1], True
        if isinstance(x, list) and len(x) == 2 and isinstance(x[1], torch.Tensor) and x[1].dim() == 1:
            return x[0], x[1], True
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        return x, torch.arange(x.size(0), device=x.device), False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all filter specs and return ``(B, C * specs, H, W)``.

        Args:
            x: Input tensor shaped ``(B, C, H, W)`` or the cached-loader form
                ``(x, idx)`` produced by ``build_dataloader_cached``.

        Returns:
            Tensor with one output channel per input channel and filter spec.
        """
        x, idx, use_cache = self._batch_input(x)
        assert x.dim() == 4, f"expected (B, C, H, W), got {tuple(x.shape)}"
        B, C, H, W = x.shape
        assert C == self.in_channels, f"in_channels={self.in_channels}, input C={C}"

        out_dtype = next(self.parameters()).dtype
        out = torch.empty((B, self.out_channels, H, W), dtype=out_dtype, device=self.device)
        for b in range(B):
            for c in range(C):
                base_key = f"{int(idx[b])}_{c}"
                direct_payloads = {}
                for spec in self.filter_specs:
                    info, norm_attrs, valuation_increments = self._get_tree_payload_for_sample(
                        base_key,
                        x[b, c],
                        spec,
                        direct_payloads,
                        use_cache=use_cache,
                    )

                    y_out = self._apply_spec(
                        spec,
                        info,
                        norm_attrs,
                        valuation_increments,
                        self.beta_f,
                    )
                    out[b, c * self.num_specs + spec.index].copy_(y_out, non_blocking=True)
        return out

    def monotonicity_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Return the per-spec monotonicity regularization for CFP gates.

        Specs with ``monotonicity_weight=0.0`` are skipped. Callers must add
        the returned scalar to their training objective explicitly; this method
        never changes ``forward`` outputs or legacy training code.
        """
        x, idx, use_cache = self._batch_input(x)
        assert x.dim() == 4, f"expected (B, C, H, W), got {tuple(x.shape)}"
        B, C, _, _ = x.shape
        assert C == self.in_channels, f"in_channels={self.in_channels}, input C={C}"

        active_specs = [spec for spec in self.filter_specs if spec.monotonicity_weight > 0.0]
        if not active_specs:
            return self._zero_parameter_penalty()
        num_applications = B * C
        if num_applications == 0:
            return self._zero_parameter_penalty()

        penalty = self._zero_parameter_penalty()
        for b in range(B):
            for c in range(C):
                base_key = f"{int(idx[b])}_{c}"
                direct_payloads = {}
                for spec in active_specs:
                    info, norm_attrs, _ = self._get_tree_payload_for_sample(
                        base_key,
                        x[b, c],
                        spec,
                        direct_payloads,
                        use_cache=use_cache,
                    )
                    penalty = penalty + self._monotonicity_penalty_for_spec(spec, info, norm_attrs)
        return penalty / float(num_applications)

    def predict(self, x: torch.Tensor, beta_f: float = 1000.0) -> torch.Tensor:
        """Run inference with a caller-provided sigmoid gain.

        The method temporarily switches the module to evaluation mode, runs
        ``forward`` under ``torch.no_grad()``, restores ``beta_f``, and restores
        the previous training/eval state.
        """
        was_training = self.training
        self.eval()
        old_beta = self.beta_f
        self.beta_f = float(beta_f)
        try:
            with torch.no_grad():
                result = self.forward(x)
        finally:
            self.beta_f = old_beta
            self.train(was_training)
        return result

    def inspect_training_sample(self, img: torch.Tensor, channel: int = 0, idx: int | None = None, build_if_missing: bool = True):
        """Return cached or direct attributes, valuations, and parameters per spec.

        Args:
            img: Image tensor shaped ``(H, W)`` or ``(C, H, W)``.
            channel: Channel to inspect when ``img`` has multiple channels.
            idx: Optional stable dataset index used to look up cached payloads.
            build_if_missing: Build a temporary tree payload when no cache entry
                exists for ``idx``.

        Returns:
            Dictionary keyed by filter-spec name. Each entry contains raw and
            normalized attributes, valuation increments, and current trainable
            parameters.
        """
        if img.dim() == 2:
            imgCHW = img.unsqueeze(0)
        elif img.dim() == 3:
            imgCHW = img
        else:
            raise ValueError(f"img must be (H, W) or (C, H, W); got {tuple(img.shape)}")

        C, _, _ = imgCHW.shape
        if C != self.in_channels and C != 1:
            raise AssertionError(f"in_channels={self.in_channels}, input C={C}")
        c = channel if C > 1 else 0

        payloads = {}
        if idx is not None:
            base_key = f"{idx}_{c}"
            for spec in self.filter_specs:
                if build_if_missing:
                    self._ensure_tree_payload_cached(base_key, imgCHW[c], spec.tree_key)
                elif base_key not in self._tree_info or spec.tree_key not in self._tree_info[base_key]:
                    raise KeyError("Tree/attributes not found in cache. Use build_if_missing=True.")
            self._maybe_refresh_norm_for_key(base_key)
            for tree_key in self._tree_info.get(base_key, {}):
                payloads[tree_key] = {
                    "info": self._tree_info[base_key][tree_key],
                    "base_attrs": self._base_attrs[base_key][tree_key],
                    "norm_attrs": self._norm_attrs[base_key][tree_key],
                    "valuation_increments": self._valuation_increments[base_key][tree_key],
                }
        else:
            img_np = self._to_numpy_u8(imgCHW[c].detach())
            for tree_key in self._tree_spec_by_key:
                payloads[tree_key] = self._compute_tree_payload(img_np, tree_key, update_stats=False)

        specs = {}
        for spec in self.filter_specs:
            payload = payloads[spec.tree_key]
            cols_raw = [payload["base_attrs"][attr_type].view(-1, 1) for attr_type in spec.attributes]
            cols_norm = [payload["norm_attrs"][attr_type].view(-1, 1) for attr_type in spec.attributes]
            specs[spec.key] = {
                "tree_type": spec.tree_type,
                "attributes": spec.attributes,
                "valuation": spec.valuation,
                "base_attrs": torch.cat(cols_raw, dim=1),
                "norm_attrs": torch.cat(cols_norm, dim=1),
                "valuation_increments": payload["valuation_increments"][spec.valuation_key],
                "weight": self._weights[spec.key],
                "bias": self._biases[spec.key],
            }
        return {"specs": specs}

    def freeze_ds_stats(self):
        """Stop updating dataset-level normalization statistics."""
        self._stats_frozen = True

    def unfreeze_ds_stats(self):
        """Resume updating dataset-level normalization statistics."""
        self._stats_frozen = False

    def refresh_cached_normalization(self):
        """Recompute normalized attributes for all cached samples."""
        for base_key in list(self._base_attrs.keys()):
            self._norm_epoch_by_key[base_key] = -1
            self._maybe_refresh_norm_for_key(base_key)

    def save_stats(self, path: str):
        """Save dataset-level normalization statistics.

        The payload is a torch-safe dictionary containing a format version,
        ``scale_mode``, and serialized dataset statistics. Per-sample caches are
        not saved.
        """
        payload = {
            "format_version": 3,
            "scale_mode": self.scale_mode,
            "ds_stats": self._serialize_ds_stats(),
        }
        torch.save(payload, path)

    def load_stats(self, path: str, refresh_cache: bool = True):
        """Load dataset-level normalization statistics.

        Args:
            path: File previously written by ``save_stats``.
            refresh_cache: Whether to recompute normalized cached attributes
                immediately after loading.
        """
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self._ds_stats = self._deserialize_ds_stats(payload.get("ds_stats", {}))
        self._stats_epoch += 1
        if refresh_cache:
            self.refresh_cached_normalization()

    def get_config(self) -> dict[str, Any]:
        """Return the architecture/configuration needed to reconstruct the layer.

        The returned dictionary is serializable and accepted by
        ``from_config``. It describes layer structure, filter specs, valuation
        choices, normalization mode, sigmoid gain, clamp bounds, and hybrid
        normalization constants. It does not include trainable weights or
        dataset statistics.
        """
        return {
            "in_channels": self.in_channels,
            "filter_specs": [self._serialize_filter_spec_config(spec) for spec in self.filter_specs],
            "scale_mode": self.scale_mode,
            "eps": self.eps,
            "beta_f": self.beta_f,
            "clamp": None if self.clamp is None else list(self.clamp),
            "hybrid_k": self.hybrid_k,
            "hybrid_floor_a": self.hybrid_floor_a,
            "attribute_dtype": self.attribute_dtype.name,
        }

    def get_weight_contract(self) -> dict[str, Any]:
        """Return the CFP contract that defines parameter names and semantics.

        Checkpoints use this contract to reject incompatible layer
        architectures before loading state into differently shaped CFP
        parameters.
        """
        return {
            "in_channels": self.in_channels,
            "filter_specs": [self._serialize_filter_spec_config(spec) for spec in self.filter_specs],
            "scale_mode": self.scale_mode,
            "eps": self.eps,
            "beta_f": self.beta_f,
            "clamp": None if self.clamp is None else list(self.clamp),
            "hybrid_k": self.hybrid_k,
            "hybrid_floor_a": self.hybrid_floor_a,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, device=None) -> "ConnectedFilterPreprocessingLayer":
        """Reconstruct a layer from ``get_config()`` output."""
        kwargs = cls._deserialize_config(config)
        if device is not None:
            kwargs["device"] = device
        return cls(**kwargs)

    def get_extra_state(self) -> dict[str, Any]:
        """Embed persistent CFP state in PyTorch checkpoints.

        This includes the weight contract and dataset normalization statistics.
        Per-sample tree/attribute caches are intentionally not persisted.
        """
        return {
            "weight_contract": self.get_weight_contract(),
            "ds_stats": self._serialize_ds_stats(),
            "stats_epoch": int(self._stats_epoch),
            "stats_frozen": bool(self._stats_frozen),
        }

    def set_extra_state(self, state: Any) -> None:
        """Restore persistent CFP state from ``state_dict`` and validate compatibility."""
        if state is None:
            return
        if not isinstance(state, Mapping):
            raise TypeError("ConnectedFilterPreprocessingLayer extra state must be a mapping.")

        saved_contract = state.get("weight_contract", None)
        if saved_contract is None and "config" in state:
            saved_contract = state["config"]
        if saved_contract is not None and self._canonical_contract(saved_contract) != self.get_weight_contract():
            raise RuntimeError(
                "ConnectedFilterPreprocessingLayer checkpoint weight contract is incompatible "
                "with the current layer. Recreate the layer with ConnectedFilterPreprocessingLayer.from_config(...)."
            )

        self._ds_stats = self._deserialize_ds_stats(state.get("ds_stats", {}))
        self._stats_epoch = int(state.get("stats_epoch", self._stats_epoch + 1))
        self._stats_frozen = bool(state.get("stats_frozen", self._stats_frozen))
        self.refresh_cached_normalization()

    def export_params(self, path: str):
        """Export CFP parameters and metadata for inspection.

        This is not the recommended training checkpoint API. Use
        ``mtlearn.layers.save_checkpoint`` for full PyTorch models.
        """
        torch.save(
            {
                "weights": {name: p.detach().cpu() for name, p in self._weights.items()},
                "biases": {name: p.detach().cpu() for name, p in self._biases.items()},
                "scale_mode": self.scale_mode,
                "clamp": None if self.clamp is None else list(self.clamp),
                "config": self.get_config(),
                "weight_contract": self.get_weight_contract(),
                "filter_specs": [self._serialize_filter_spec(spec) for spec in self.filter_specs],
            },
            path,
        )

    def save_params(self, path: str):
        """Compatibility alias for ``export_params``."""
        self.export_params(path)

    @staticmethod
    def _attribute_from_name(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        for enum_type in (morphology.AttributeType, morphology.AttributeGroup):
            try:
                return getattr(enum_type, value)
            except AttributeError:
                pass
        raise ValueError(f"unknown CFP attribute or group name: {value}")

    @staticmethod
    def _tos_interpolation_from_name(value: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        try:
            return getattr(morphology.ToSInterpolation, value)
        except AttributeError as exc:
            raise ValueError(f"unknown tree-of-shapes interpolation name: {value}") from exc

    @classmethod
    def _valuation_from_config(cls, value: Any) -> CFPValuation:
        if value is None:
            return CFPValuation.ALTITUDE
        if isinstance(value, CFPValuation):
            return value
        if isinstance(value, str):
            if value == "altitude":
                return CFPValuation.ALTITUDE
            if value == "altitude_tophat":
                return CFPValuation.ALTITUDE_TOPHAT
            return CFPValuation.node_attribute(cls._attribute_from_name(value))
        if not isinstance(value, Mapping):
            return CFPValuation.node_attribute(cls._attribute_from_name(value))

        kind = value.get("kind", "altitude")
        if kind == "altitude":
            return CFPValuation.ALTITUDE
        if kind == "altitude_tophat":
            return CFPValuation.ALTITUDE_TOPHAT
        if kind == "node_attribute":
            return CFPValuation.node_attribute(cls._attribute_from_name(value.get("attribute")))
        raise ValueError(f"unknown CFP valuation kind in config: {kind!r}")

    @classmethod
    def _deserialize_filter_spec_config(cls, spec: Mapping[str, Any]) -> dict[str, Any]:
        if "tree_type" not in spec:
            raise ValueError("serialized filter spec is missing tree_type.")
        if "attributes" not in spec:
            raise ValueError("serialized filter spec is missing attributes.")

        restored = {
            "tree_type": spec["tree_type"],
            "attributes": tuple(cls._attribute_from_name(attr) for attr in spec["attributes"]),
        }
        if "name" in spec:
            restored["name"] = spec["name"]
        if "valuation" in spec:
            restored["valuation"] = cls._valuation_from_config(spec["valuation"])
        if "preserve_root" in spec:
            restored["preserve_root"] = bool(spec["preserve_root"])
        if "monotonicity_weight" in spec:
            restored["monotonicity_weight"] = _normalize_nonnegative_scalar(
                spec["monotonicity_weight"],
                "monotonicity_weight",
            )
        tos_interpolation = spec.get("tos_interpolation", None)
        if tos_interpolation is not None:
            restored["tos_interpolation"] = cls._tos_interpolation_from_name(tos_interpolation)
        if "tos_infinity_seed_row" in spec:
            restored["tos_infinity_seed_row"] = int(spec["tos_infinity_seed_row"])
        if "tos_infinity_seed_col" in spec:
            restored["tos_infinity_seed_col"] = int(spec["tos_infinity_seed_col"])
        return restored

    @classmethod
    def _deserialize_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TypeError("ConnectedFilterPreprocessingLayer config must be a mapping.")
        if "config" in config and "filter_specs" not in config:
            config = config["config"]

        return {
            "in_channels": int(config["in_channels"]),
            "filter_specs": [
                cls._deserialize_filter_spec_config(spec)
                for spec in config["filter_specs"]
            ],
            "scale_mode": config.get("scale_mode", "hybrid"),
            "eps": float(config.get("eps", 1e-6)),
            "beta_f": float(config.get("beta_f", 1.0)),
            "clamp": config.get("clamp", None),
            "hybrid_k": float(config.get("hybrid_k", 3.0)),
            "hybrid_floor_a": float(config.get("hybrid_floor_a", 0.05)),
            "attribute_dtype": config.get("attribute_dtype", None),
        }

    @classmethod
    def _canonical_contract(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        return cls(**cls._deserialize_config(config)).get_weight_contract()

    def _serialize_ds_stats(self) -> dict[str, dict[str, Any]]:
        return {
            str(key): {
                name: value.detach().cpu() if torch.is_tensor(value) else value
                for name, value in stats.items()
            }
            for key, stats in self._ds_stats.items()
        }

    def _deserialize_ds_stats(self, serialized: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            str(key): {
                name: value.to(self.device) if torch.is_tensor(value) else value
                for name, value in stats.items()
            }
            for key, stats in serialized.items()
        }

    @staticmethod
    def _serialize_filter_spec_config(spec: _NormalizedFilterSpec) -> dict[str, Any]:
        tos_interpolation = None if spec.tos_interpolation is None else _enum_name(spec.tos_interpolation)
        spec_config = {
            "name": spec.key,
            "tree_type": spec.tree_type,
            "attributes": [_enum_name(attr) for attr in spec.attributes],
            "valuation": {
                "kind": spec.valuation.kind,
                "attribute": None if _uses_altitude_signal(spec.valuation) else _enum_name(spec.valuation.attribute),
            },
            "preserve_root": spec.preserve_root,
            "monotonicity_weight": spec.monotonicity_weight,
            "tos_interpolation": tos_interpolation,
            "tos_infinity_seed_row": spec.tos_infinity_seed_row,
            "tos_infinity_seed_col": spec.tos_infinity_seed_col,
        }
        return spec_config

    @staticmethod
    def _serialize_filter_spec(spec: _NormalizedFilterSpec) -> dict[str, Any]:
        tos_interpolation = None if spec.tos_interpolation is None else _enum_name(spec.tos_interpolation)
        valuation_attribute = "ALTITUDE" if _uses_altitude_signal(spec.valuation) else _enum_name(spec.valuation.attribute)
        return {
            "index": spec.index,
            "key": spec.key,
            "name": spec.key,
            "tree_type": spec.tree_type,
            "tree_key": spec.tree_key,
            "attributes": [_enum_name(attr) for attr in spec.attributes],
            "valuation": {
                "kind": spec.valuation.kind,
                "attribute": valuation_attribute,
            },
            "valuation_key": spec.valuation_key,
            "preserve_root": spec.preserve_root,
            "monotonicity_weight": spec.monotonicity_weight,
            "tos_interpolation": tos_interpolation,
            "tos_infinity_seed_row": spec.tos_infinity_seed_row,
            "tos_infinity_seed_col": spec.tos_infinity_seed_col,
        }

    def get_params(self):
        """Return CPU clones of the per-spec weight and bias tensors."""
        return (
            {name: p.detach().cpu().clone() for name, p in self._weights.items()},
            {name: p.detach().cpu().clone() for name, p in self._biases.items()},
        )

    @staticmethod
    def _logit(p: float) -> float:
        p = max(min(float(p), 1.0 - 1e-6), 1e-6)
        return math.log(p / (1.0 - p))

    @torch.no_grad()
    def init_identity_with_bias(self, p0: float = 0.995):
        """Initialize filters close to identity using only positive bias.

        Weights are set to zero and each bias is chosen so the initial sigmoid
        value is approximately ``p0``.
        """
        L = self._logit(p0) / float(self.beta_f)
        for spec in self.filter_specs:
            self._weights[spec.key].zero_()
            self._biases[spec.key].fill_(L)

    @torch.no_grad()
    def init_identity_bias_zero(self, p0: float = 0.99):
        """Initialize filters close to identity with zero bias.

        This initialization assumes hybrid-normalized attributes with a
        positive floor. Each weight receives the same positive value and biases
        are set to zero.
        """
        a = max(min(self.hybrid_floor_a, 1.0), 1e-6)
        L = self._logit(p0) / float(self.beta_f)
        for spec in self.filter_specs:
            c = L / (len(spec.attributes) * a)
            self._weights[spec.key].fill_(c)
            self._biases[spec.key].zero_()

    def build_dataloader_cached(self, dataloader):
        """Wrap a DataLoader and precompute CFP caches/statistics.

        The returned DataLoader yields ``((x, idx), y)`` batches with stable
        dataset indices. During the prepass, this layer builds tree payloads and
        updates dataset-level statistics for every sample/channel/tree key.
        Statistics are frozen and cached normalizations are refreshed before
        the wrapped loader is returned.
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

        self._stats_frozen = False
        with torch.no_grad():
            for (x, idx), _ in new_loader:
                B, C, _, _ = x.shape
                for b in range(B):
                    for c in range(C):
                        base_key = f"{int(idx[b])}_{c}"
                        for tree_key in self._tree_spec_by_key:
                            self._ensure_tree_payload_cached(base_key, x[b, c], tree_key)

        self.freeze_ds_stats()
        self.refresh_cached_normalization()
        return new_loader

    def build_dataloader_cached_fixed_stats(self, dataloader, *, index_offset: int = 0):
        """Wrap a DataLoader and precompute CFP caches without updating stats.

        Use this for validation/test splits after training statistics have been
        built with ``build_dataloader_cached(...)`` or restored with
        ``load_stats(...)``. The returned DataLoader yields
        ``((x, idx + index_offset), y)`` so callers can keep split cache keys
        disjoint.
        """
        from torch.utils.data import DataLoader

        self._require_fixed_dataset_stats()

        dataset_wrapped = IndexedDatasetWrapper(dataloader.dataset, index_offset=index_offset)
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

        with torch.no_grad():
            for (x, idx), _ in new_loader:
                B, C, _, _ = x.shape
                for b in range(B):
                    for c in range(C):
                        base_key = f"{int(idx[b])}_{c}"
                        for tree_key in self._tree_spec_by_key:
                            self._ensure_tree_payload_cached(
                                base_key,
                                x[b, c],
                                tree_key,
                                update_stats=False,
                            )

        return new_loader


CFPLayer = ConnectedFilterPreprocessingLayer

__all__ = [
    'CFPValuation',
    'ConnectedFilterPreprocessingImplicitJacobianFunction',
    'ConnectedFilterPreprocessingLayer',
    'CFPLayer',
]
