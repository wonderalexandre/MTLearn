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

import math
import numbers
from typing import Any, Mapping
import torch
import numpy as np
from .._helpers import (
    to_numpy_u8,
)
from .normalization import AttributeNormalizer, StatsSerializer
from .runtime import (
    BatchInputNormalizer,
    CachedDataLoaderBuilder,
    CFPContext,
    ConnectedFilterPreprocessingImplicitJacobianFunction,
    ForwardExecutor,
    TrainingSampleInspector,
    TreePayloadCache,
    TreePayloadProvider,
    TreeReconstructionFunction,
    TreeReconstructor,
)
from .scoring import LegacyLinearParameterInitializer
from .serialization import ConfigDeserializer, ConfigSerializer, PersistentStateManager
from .specs import NormalizedFilterSpec as _NormalizedFilterSpec
from .valuation import CFPValuation, ValuationProjection
from .component_registries import (
    constraint_configs_for_spec,
    create_regularizer,
    create_score_constraint,
    regularizer_configs_for_spec,
)
from .specs.filter_spec_normalizer import (
    enum_name,
    filter_spec_tree_key,
    normalize_filter_specs,
    normalize_nonnegative_scalar,
    validate_valuation_for_tree_type,
)


def _enum_name(value: Any) -> str:
    return enum_name(value)


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
    return normalize_nonnegative_scalar(value, name)


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
        self._scoring_models = torch.nn.ModuleDict(
            {spec.key: spec.scoring_model for spec in self.filter_specs}
        ).to(self.device)
        self._score_constraints = torch.nn.ModuleDict(
            {
                spec.key: torch.nn.ModuleList(
                    [
                        create_score_constraint(config)
                        for config in constraint_configs_for_spec(spec)
                    ]
                )
                for spec in self.filter_specs
            }
        )
        self._regularizers = torch.nn.ModuleDict(
            {
                spec.key: torch.nn.ModuleList(
                    [
                        create_regularizer(config)
                        for config in regularizer_configs_for_spec(spec)
                    ]
                )
                for spec in self.filter_specs
            }
        )

        self._spec_by_key = {spec.key: spec for spec in self.filter_specs}
        self._tree_spec_by_key = {}
        self._scoring_attrs_by_tree_key = {}
        self._valuation_projections_by_tree_key = {}
        for spec in self.filter_specs:
            self._tree_spec_by_key.setdefault(spec.tree_key, spec)
            self._scoring_attrs_by_tree_key.setdefault(spec.tree_key, set()).update(spec.attributes)
            self._valuation_projections_by_tree_key.setdefault(spec.tree_key, {})[
                spec.valuation_key
            ] = spec.valuation_projection

        self._attribute_normalizer = AttributeNormalizer(
            self.scale_mode,
            self.eps,
            hybrid_k=self.hybrid_k,
            hybrid_floor_a=self.hybrid_floor_a,
        )
        self._tree_payload_provider = TreePayloadProvider(
            tree_spec_by_key=self._tree_spec_by_key,
            scoring_attrs_by_tree_key=self._scoring_attrs_by_tree_key,
            valuation_projections_by_tree_key=self._valuation_projections_by_tree_key,
            normalizer=self._attribute_normalizer,
            stat_key_fn=self._stat_key,
            device=self.device,
            attribute_dtype=self.attribute_dtype,
        )
        self._tree_payload_cache = TreePayloadCache()
        self._batch_input_normalizer = BatchInputNormalizer()
        self._dataloader_cache_builder = CachedDataLoaderBuilder()
        self._forward_executor = ForwardExecutor()
        self._training_sample_inspector = TrainingSampleInspector()
        self._config_serializer = ConfigSerializer()
        self._stats_serializer = StatsSerializer()
        self._legacy_parameter_initializer = LegacyLinearParameterInitializer()
        self._persistent_state_manager = PersistentStateManager()
        self._active_context = None

        self._weights, self._biases = self._legacy_parameter_initializer.create_parameter_dicts(
            self.filter_specs,
            device=self.device,
        )

    @staticmethod
    def _tree_key(tree_type, tos_interpolation, tos_infinity_seed_row, tos_infinity_seed_col) -> str:
        return filter_spec_tree_key(
            tree_type,
            tos_interpolation,
            tos_infinity_seed_row,
            tos_infinity_seed_col,
        )

    def _normalize_filter_specs(
        self,
        filter_specs,
        *,
        default_tos_interpolation,
        default_tos_infinity_seed_row: int,
        default_tos_infinity_seed_col: int,
    ):
        return normalize_filter_specs(
            filter_specs,
            default_tos_interpolation=default_tos_interpolation,
            default_tos_infinity_seed_row=default_tos_infinity_seed_row,
            default_tos_infinity_seed_col=default_tos_infinity_seed_col,
        )

    @staticmethod
    def _validate_valuation_for_tree_type(valuation: CFPValuation, tree_type: str) -> None:
        validate_valuation_for_tree_type(valuation, tree_type)

    def _stat_key(self, tree_key: str, attr_type: Any) -> str:
        return f"{tree_key}::{_enum_name(attr_type)}"

    @property
    def _ds_stats(self):
        return self._attribute_normalizer.ds_stats

    @_ds_stats.setter
    def _ds_stats(self, value):
        self._attribute_normalizer.ds_stats = value

    @property
    def _stats_epoch(self) -> int:
        return self._attribute_normalizer.stats_epoch

    @_stats_epoch.setter
    def _stats_epoch(self, value: int) -> None:
        self._attribute_normalizer.stats_epoch = int(value)

    @property
    def _stats_frozen(self) -> bool:
        return self._attribute_normalizer.stats_frozen

    @_stats_frozen.setter
    def _stats_frozen(self, value: bool) -> None:
        self._attribute_normalizer.stats_frozen = bool(value)

    @property
    def _tree_info(self):
        return self._tree_payload_cache.tree_info()

    @property
    def _base_attrs(self):
        return self._tree_payload_cache.base_attrs()

    @property
    def _norm_attrs(self):
        return self._tree_payload_cache.norm_attrs()

    @property
    def _valuation_increments(self):
        return self._tree_payload_cache.valuation_increments()

    @property
    def _norm_epoch_by_key(self):
        return self._tree_payload_cache.norm_epoch_by_key

    def _to_numpy_u8(self, img2d_t: torch.Tensor) -> np.ndarray:
        return to_numpy_u8(img2d_t)

    def _build_tree(self, img_np: np.ndarray, spec: _NormalizedFilterSpec):
        return self._tree_payload_provider.build_tree(img_np, spec)

    def _compute_tree_info(self, tree, spec: _NormalizedFilterSpec):
        return self._tree_payload_provider.compute_tree_info(tree, spec)

    def _update_ds_stats(self, stat_key, a_raw_1d: torch.Tensor):
        self._attribute_normalizer.update(stat_key, a_raw_1d)

    def _normalize_with_ds_stats(self, stat_key, a_raw_1d: torch.Tensor) -> torch.Tensor:
        return self._attribute_normalizer.normalize(stat_key, a_raw_1d)

    def _compute_valuation_increment(self, tree, info, valuation: ValuationProjection) -> torch.Tensor:
        return self._tree_payload_provider.compute_valuation_increment(tree, info, valuation)

    def _compute_tree_payload(self, img_np: np.ndarray, tree_key: str, *, update_stats: bool):
        return self._tree_payload_provider.compute_payload(img_np, tree_key, update_stats=update_stats)

    def _ensure_tree_payload_cached(
        self,
        base_key: str,
        img_t: torch.Tensor,
        tree_key: str,
        *,
        update_stats: bool = True,
    ) -> None:
        if self._tree_payload_cache.has(base_key, tree_key):
            return

        img_np = self._to_numpy_u8(img_t.detach())
        payload = self._compute_tree_payload(img_np, tree_key, update_stats=update_stats)
        self._tree_payload_cache.set(base_key, tree_key, payload)
        self._tree_payload_cache.set_epoch(base_key, self._stats_epoch)

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
        cached_payloads = self._tree_payload_cache.sample_payloads(base_key)
        if not cached_payloads:
            return
        if self._norm_epoch_by_key.get(base_key, -1) == self._stats_epoch:
            return

        refreshed = {}
        for tree_key, payload in cached_payloads.items():
            refreshed[tree_key] = {}
            per_attr_raw = payload["base_attrs"]
            for attr_type, a_raw_2d in per_attr_raw.items():
                stat_key = self._stat_key(tree_key, attr_type)
                refreshed[tree_key][attr_type] = self._normalize_with_ds_stats(stat_key, a_raw_2d.view(-1))
        self._tree_payload_cache.replace_norm_attrs(base_key, refreshed)
        self._tree_payload_cache.set_epoch(base_key, self._stats_epoch)

    def _normalized_attribute_matrix(self, spec: _NormalizedFilterSpec, norm_attrs, *, dtype: torch.dtype):
        cols = [norm_attrs[attr_type].view(-1, 1).to(dtype=dtype, device=self.device) for attr_type in spec.attributes]
        return torch.cat(cols, dim=1)

    def _module_dtype(self) -> torch.dtype:
        for parameter in self.parameters():
            return parameter.dtype
        return torch.float32

    def _score_dtype(self, spec: _NormalizedFilterSpec) -> torch.dtype:
        if spec.key in self._weights:
            return self._weights[spec.key].dtype
        for parameter in self._scoring_models[spec.key].parameters():
            return parameter.dtype
        return self._module_dtype()

    def _context_for(self, base_key: str, batch_index: int, channel_index: int, spec: _NormalizedFilterSpec, *, mode: str) -> CFPContext:
        return CFPContext(
            sample_key=base_key,
            batch_index=batch_index,
            channel_index=channel_index,
            spec_name=spec.key,
            extras={"mode": mode},
        )

    def _score_nodes(self, spec: _NormalizedFilterSpec, info, norm_attrs, beta_f: float, context: CFPContext | None = None) -> torch.Tensor:
        dtype = self._score_dtype(spec)
        A_norm = self._normalized_attribute_matrix(spec, norm_attrs, dtype=dtype)
        scoring_model = self._scoring_models[spec.key]
        if spec.key in self._weights:
            scores = scoring_model(
                A_norm,
                info,
                context=context,
                weight=self._weights[spec.key],
                bias=self._biases[spec.key],
                beta_f=beta_f,
                clamp=self.clamp,
            )
        else:
            scores = scoring_model(A_norm, info, context=context, beta_f=beta_f, clamp=self.clamp)
        scores = scores.view(-1)
        num_nodes = int(info["tpre"].numel())
        if scores.numel() != num_nodes:
            raise ValueError(f"scoring model returned {scores.numel()} scores for {num_nodes} tree nodes.")
        for constraint in self._score_constraints[spec.key]:
            scores = constraint(scores, info, context=context)
        return scores

    def _monotonicity_penalty_for_spec(self, spec: _NormalizedFilterSpec, info, norm_attrs) -> torch.Tensor:
        context = getattr(self, "_active_context", None)
        scores = self._score_nodes(spec, info, norm_attrs, self.beta_f, context=context)
        features = self._normalized_attribute_matrix(spec, norm_attrs, dtype=self._score_dtype(spec))
        penalty = self._zero_parameter_penalty()
        for regularizer in self._regularizers[spec.key]:
            penalty = penalty + regularizer(scores, info, features=features, context=context)
        return penalty

    def _zero_parameter_penalty(self) -> torch.Tensor:
        for parameter in self.parameters():
            return parameter.sum() * 0.0
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def _get_tree_payload_for_sample(self, base_key: str, img_ch: torch.Tensor, spec: _NormalizedFilterSpec, direct_payloads, *, use_cache: bool):
        if use_cache:
            self._ensure_tree_payload_cached(base_key, img_ch, spec.tree_key)
            self._maybe_refresh_norm_for_key(base_key)
            payload = self._tree_payload_cache.get(base_key, spec.tree_key)
            return payload["info"], payload["norm_attrs"], payload["valuation_increments"]

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
        dtype = self._score_dtype(spec)
        valuation_payload = {"valuation_increments": valuation_increments}
        increments = spec.valuation_projection.node_signal(valuation_payload, info).to(
            dtype=dtype,
            device=self.device,
        )
        scores = self._score_nodes(spec, info, norm_attrs, beta_f, context=getattr(self, "_active_context", None))
        filtered_increments = increments * scores

        y_ch = TreeReconstructionFunction.apply(
            filtered_increments,
            info["tpre"],
            info["tpost"],
            info["parent"],
            info["node_of_pixel"],
            info["numRows"],
            info["numCols"],
            info["order_forward"],
            info["order_backward"],
        )

        base = None
        if spec.valuation_projection.requires_unfiltered_image():
            base = TreeReconstructor.forward_from_info(increments, info)
        return spec.valuation_projection.project(y_ch, base, info)

    def _batch_input(self, x):
        return self._batch_input_normalizer.normalize(x).as_tuple()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all filter specs and return ``(B, C * specs, H, W)``.

        Args:
            x: Input tensor shaped ``(B, C, H, W)`` or the cached-loader form
                ``(x, idx)`` produced by ``build_dataloader_cached``.

        Returns:
            Tensor with one output channel per input channel and filter spec.
        """
        return self._forward_executor.forward(self, x)

    def monotonicity_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Return the per-spec monotone-score regularization.

        Specs with ``monotonicity_weight=0.0`` are skipped. Callers must add
        this scalar to their training objective explicitly.
        """
        return self._forward_executor.monotonicity_penalty(self, x)

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
        return self._training_sample_inspector.inspect(
            self,
            img,
            channel=channel,
            idx=idx,
            build_if_missing=build_if_missing,
        )

    def freeze_ds_stats(self):
        """Stop updating dataset-level normalization statistics."""
        self._stats_frozen = True

    def unfreeze_ds_stats(self):
        """Resume updating dataset-level normalization statistics."""
        self._stats_frozen = False

    def refresh_cached_normalization(self):
        """Recompute normalized attributes for all cached samples."""
        for base_key in list(self._tree_payload_cache.sample_keys()):
            self._tree_payload_cache.invalidate_sample_normalization(base_key)
            self._maybe_refresh_norm_for_key(base_key)

    def save_stats(self, path: str):
        """Save dataset-level normalization statistics.

        The payload is a torch-safe dictionary containing a format version,
        ``scale_mode``, and serialized dataset statistics. Per-sample caches are
        not saved.
        """
        self._persistent_state_manager.save_stats(self, path)

    def load_stats(self, path: str, refresh_cache: bool = True):
        """Load dataset-level normalization statistics.

        Args:
            path: File previously written by ``save_stats``.
            refresh_cache: Whether to recompute normalized cached attributes
                immediately after loading.
        """
        self._persistent_state_manager.load_stats(
            self,
            path,
            refresh_cache=refresh_cache,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the architecture/configuration needed to reconstruct the layer.

        The returned dictionary is serializable and accepted by
        ``from_config``. It describes layer structure, filter specs, valuation
        choices, normalization mode, sigmoid gain, clamp bounds, and hybrid
        normalization constants. It does not include trainable weights or
        dataset statistics.
        """
        return self._config_serializer.layer_config(self)

    def get_parameter_contract(self) -> dict[str, Any]:
        """Return parameter names and shapes owned by this CFP layer."""
        return self._config_serializer.parameter_contract(self)

    def get_inference_contract(self) -> dict[str, Any]:
        """Return the CFP contract that defines forward/inference semantics."""
        return self._config_serializer.inference_contract(self)

    def get_training_contract(self) -> dict[str, Any]:
        """Return training-only CFP settings such as regularization weights."""
        return self._config_serializer.training_contract(self)

    def get_contracts(self) -> dict[str, Any]:
        """Return named CFP contracts for parameters, inference, and training."""
        return self._config_serializer.contracts(self)

    def get_weight_contract(self) -> dict[str, Any]:
        """Compatibility alias for the inference contract used by checkpoints."""
        return self.get_inference_contract()

    @staticmethod
    def _training_contract_for_spec(spec: _NormalizedFilterSpec) -> dict[str, Any]:
        return ConfigSerializer.training_contract_for_spec(spec)

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
        return self._persistent_state_manager.extra_state(self)

    def set_extra_state(self, state: Any) -> None:
        """Restore persistent CFP state from ``state_dict`` and validate compatibility."""
        self._persistent_state_manager.set_extra_state(self, state)

    def export_params(self, path: str):
        """Export CFP parameters and metadata for inspection.

        This is not the recommended training checkpoint API. Use
        ``mtlearn.layers.save_checkpoint`` for full PyTorch models.
        """
        self._persistent_state_manager.export_params(self, path)

    def save_params(self, path: str):
        """Compatibility alias for ``export_params``."""
        self.export_params(path)

    @staticmethod
    def _attribute_from_name(value: Any) -> Any:
        return ConfigDeserializer.attribute_from_name(value)

    @staticmethod
    def _tos_interpolation_from_name(value: Any) -> Any:
        return ConfigDeserializer.tos_interpolation_from_name(value)

    @classmethod
    def _valuation_from_config(cls, value: Any) -> CFPValuation:
        return ConfigDeserializer.valuation_from_config(value)

    @classmethod
    def _deserialize_filter_spec_config(cls, spec: Mapping[str, Any]) -> dict[str, Any]:
        return ConfigDeserializer().deserialize_filter_spec_config(spec)

    @classmethod
    def _deserialize_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        return ConfigDeserializer().deserialize_layer_config(config)

    @classmethod
    def _canonical_contract(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        return ConfigDeserializer().canonical_contract(config, layer_cls=cls)

    def _serialize_ds_stats(self) -> dict[str, dict[str, Any]]:
        return self._stats_serializer.serialize(self._ds_stats)

    def _deserialize_ds_stats(self, serialized: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        return self._stats_serializer.deserialize(serialized, device=self.device)

    @staticmethod
    def _serialize_filter_spec_config(spec: _NormalizedFilterSpec, *, include_training: bool = True) -> dict[str, Any]:
        return ConfigSerializer().filter_spec_config(spec, include_training=include_training)

    @staticmethod
    def _serialize_filter_spec(spec: _NormalizedFilterSpec) -> dict[str, Any]:
        return ConfigSerializer().filter_spec_metadata(spec)

    def get_params(self):
        """Return CPU clones of the per-spec weight and bias tensors."""
        return self._persistent_state_manager.get_params(self)

    @staticmethod
    def _logit(p: float) -> float:
        return LegacyLinearParameterInitializer.logit(p)

    @torch.no_grad()
    def init_identity_with_bias(self, p0: float = 0.995):
        """Initialize filters close to identity using only positive bias.

        Weights are set to zero and each bias is chosen so the initial sigmoid
        value is approximately ``p0``.
        """
        self._legacy_parameter_initializer.init_identity_with_bias(
            self.filter_specs,
            self._weights,
            self._biases,
            beta_f=self.beta_f,
            p0=p0,
        )

    @torch.no_grad()
    def init_identity_bias_zero(self, p0: float = 0.99):
        """Initialize filters close to identity with zero bias.

        This initialization assumes hybrid-normalized attributes with a
        positive floor. Each weight receives the same positive value and biases
        are set to zero.
        """
        self._legacy_parameter_initializer.init_identity_bias_zero(
            self.filter_specs,
            self._weights,
            self._biases,
            beta_f=self.beta_f,
            hybrid_floor_a=self.hybrid_floor_a,
            p0=p0,
        )

    def build_dataloader_cached(self, dataloader):
        """Wrap a DataLoader and precompute CFP caches/statistics.

        The returned DataLoader yields ``((x, idx), y)`` batches with stable
        dataset indices. During the prepass, this layer builds tree payloads and
        updates dataset-level statistics for every sample/channel/tree key.
        Statistics are frozen and cached normalizations are refreshed before
        the wrapped loader is returned.
        """
        return self._dataloader_cache_builder.build_cached(self, dataloader)

    def build_dataloader_cached_fixed_stats(self, dataloader, *, index_offset: int = 0):
        """Wrap a DataLoader and precompute CFP caches without updating stats.

        Use this for validation/test splits after training statistics have been
        built with ``build_dataloader_cached(...)`` or restored with
        ``load_stats(...)``. The returned DataLoader yields
        ``((x, idx + index_offset), y)`` so callers can keep split cache keys
        disjoint.
        """
        return self._dataloader_cache_builder.build_fixed_stats(
            self,
            dataloader,
            index_offset=index_offset,
        )


CFPLayer = ConnectedFilterPreprocessingLayer

__all__ = [
    'CFPValuation',
    'ConnectedFilterPreprocessingImplicitJacobianFunction',
    'ConnectedFilterPreprocessingLayer',
    'CFPLayer',
]
