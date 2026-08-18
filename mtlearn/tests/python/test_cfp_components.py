import pytest

import mtlearn

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    pytest.skip(f"PyTorch unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology
from mtlearn.layers import cfp
from mtlearn.layers.cfp.normalization import AttributeNormalizer, StatsSerializer
from mtlearn.layers.cfp.runtime import (
    BatchInputNormalizer,
    CachedDataLoaderBuilder,
    CFPCacheInputError,
    CFPContext,
    TreePayloadCache,
    TreeReconstructor,
    validate_cfp_cache_batch_x,
)
from mtlearn.layers.cfp.serialization import ConfigDeserializer, ConfigSerializer
from mtlearn.layers.cfp.specs.filter_spec_normalizer import normalize_filter_specs

pytestmark = pytest.mark.integration


def test_grouped_cfp_subpackages_preserve_public_component_identity():
    import mtlearn.layers.cfp.normalization as normalization

    from mtlearn.layers.cfp.constraints import PreserveRootConstraint
    from mtlearn.layers.cfp.regularization import (
        PathScoreMonotonicityRegularizer,
        AttributeOrderScoreMonotonicityRegularizer,
        EdgeScoreMonotonicityRegularizer,
    )
    from mtlearn.layers.cfp.scoring import MLPScorer
    from mtlearn.layers.cfp.specs import FilterSpec

    assert MLPScorer is cfp.MLPScorer
    assert PreserveRootConstraint is cfp.PreserveRootConstraint
    assert PathScoreMonotonicityRegularizer is cfp.PathScoreMonotonicityRegularizer
    assert AttributeOrderScoreMonotonicityRegularizer is cfp.AttributeOrderScoreMonotonicityRegularizer
    assert EdgeScoreMonotonicityRegularizer is cfp.EdgeScoreMonotonicityRegularizer
    assert FilterSpec is cfp.FilterSpec
    assert TreeReconstructor.__name__ == "TreeReconstructor"
    assert validate_cfp_cache_batch_x.__name__ == "validate_cfp_cache_batch_x"
    assert CFPCacheInputError.__name__ == "CFPCacheInputError"
    assert ConfigSerializer.__name__ == "ConfigSerializer"
    assert AttributeNormalizer.__name__ == "AttributeNormalizer"
    assert not hasattr(cfp, "AttributeNormalizer")
    assert not hasattr(cfp, "BatchInputNormalizer")
    assert not hasattr(cfp, "CFPCacheInputError")
    assert not hasattr(cfp, "ConfigSerializer")
    assert not hasattr(cfp, "TreePayloadCache")
    assert not hasattr(cfp, "TreeReconstructor")
    assert not hasattr(cfp, "validate_cfp_cache_batch_x")
    assert not hasattr(cfp.ConnectedFilterPreprocessingLayer, "get_params")
    assert not hasattr(cfp.ConnectedFilterPreprocessingLayer, "init_identity_bias_zero")
    assert not hasattr(normalization, "normalize_dataset_clipped_zscore01")
    assert not hasattr(normalization, "update_attribute_stats")
    with pytest.raises((ModuleNotFoundError, FileNotFoundError)):
        __import__("mtlearn.layers.cfp.mlp_scorer")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"scale_mode": "bad"}, "scale_mode"),
        ({"eps": 0.0}, "eps"),
        ({"eps": True}, "eps"),
        ({"clipped_zscore_radius": 0.0}, "clipped_zscore_radius"),
        ({"clipped_zscore_radius": True}, "clipped_zscore_radius"),
        ({"clipped_zscore_floor": -0.1}, "clipped_zscore_floor"),
        ({"clipped_zscore_floor": 1.1}, "clipped_zscore_floor"),
        ({"clipped_zscore_floor": True}, "clipped_zscore_floor"),
    ],
)
def test_attribute_normalizer_rejects_invalid_options(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        AttributeNormalizer(**kwargs)


def test_attribute_normalizer_updates_and_freezes_statistics():
    normalizer = AttributeNormalizer(scale_mode="dataset_minmax01")
    values = torch.tensor([2.0, 5.0, 3.0], dtype=torch.float32)

    assert normalizer.update("area", values) is True
    assert normalizer.stats_epoch == 1
    assert torch.equal(normalizer.ds_stats["area"]["amin"], torch.tensor(2.0))
    assert torch.equal(normalizer.ds_stats["area"]["amax"], torch.tensor(5.0))

    normalizer.freeze()
    assert normalizer.update("area", torch.tensor([0.0, 10.0], dtype=torch.float32)) is False
    assert normalizer.stats_epoch == 1
    assert torch.equal(normalizer.ds_stats["area"]["amin"], torch.tensor(2.0))
    assert torch.equal(normalizer.ds_stats["area"]["amax"], torch.tensor(5.0))


def test_attribute_normalizer_clipped_zscore_uses_float64_statistics_and_positive_range():
    normalizer = AttributeNormalizer(
        scale_mode="dataset_clipped_zscore01",
        clipped_zscore_radius=2.0,
        clipped_zscore_floor=0.1,
    )
    values = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)

    assert normalizer.update("area", values) is True
    assert normalizer.ds_stats["area"]["sum"].dtype == torch.float64

    normalized = normalizer.normalize("area", values)

    assert normalized.dtype == values.dtype
    assert torch.all((normalized >= 0.1) & (normalized <= 1.0))
    assert torch.isfinite(normalized).all()


def test_attribute_normalizer_clipped_zscore_supports_mps_stats():
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    normalizer = AttributeNormalizer(scale_mode="dataset_clipped_zscore01")
    values = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32, device="mps")

    assert normalizer.update("area", values) is True
    normalized = normalizer.normalize("area", values)

    assert normalizer.ds_stats["area"]["sum"].device.type == "cpu"
    assert normalizer.ds_stats["area"]["sum"].dtype == torch.float64
    assert normalized.device.type == "mps"
    assert normalized.dtype == values.dtype
    assert torch.isfinite(normalized).all()


def test_attribute_normalizer_minmax_supports_mps_stats():
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    normalizer = AttributeNormalizer(scale_mode="dataset_minmax01")
    values = torch.tensor([2.0, 5.0, 3.0], dtype=torch.float32, device="mps")

    assert normalizer.update("area", values) is True
    normalized = normalizer.normalize("area", values)

    assert normalizer.ds_stats["area"]["amin"].device.type == "cpu"
    assert normalizer.ds_stats["area"]["amax"].device.type == "cpu"
    assert normalized.device.type == "mps"
    assert normalized.dtype == values.dtype
    assert torch.isfinite(normalized).all()


@pytest.mark.parametrize("scale_mode", ["dataset_minmax01", "dataset_zscore", "dataset_clipped_zscore01"])
def test_attribute_normalizer_statistical_modes_require_offline_stats(scale_mode):
    normalizer = AttributeNormalizer(scale_mode=scale_mode)
    values = torch.tensor([7.0], dtype=torch.float32)

    with pytest.raises(RuntimeError, match=f"scale_mode='{scale_mode}' requires dataset statistics"):
        normalizer.normalize("missing", values)


def test_attribute_normalizer_none_does_not_require_stats():
    normalizer = AttributeNormalizer(scale_mode="none")
    values = torch.tensor([7.0], dtype=torch.float32)

    normalized = normalizer.normalize("missing", values)

    assert normalized is values


def test_linear_sigmoid_scorer_matches_explicit_formula():
    scorer = cfp.LinearSigmoidScorer(num_features=2, score_sharpness=2.0)
    with torch.no_grad():
        scorer.weight.copy_(torch.tensor([1.0, -1.0]))
        scorer.bias.zero_()
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

    scores = scorer(features)

    expected = torch.sigmoid(torch.tensor([2.0, -2.0, 0.0], dtype=torch.float32))
    assert torch.allclose(scores, expected)


def test_linear_sigmoid_scorer_accepts_external_parameters():
    scorer = cfp.LinearSigmoidScorer(num_features=2, score_sharpness=1.0, owns_parameters=False)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    weight = torch.tensor([2.0, -2.0], dtype=torch.float32, requires_grad=True)
    bias = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

    scores = scorer(features, weight=weight, bias=bias, score_sharpness=0.5)
    scores.sum().backward()

    assert torch.allclose(scores, torch.sigmoid(torch.tensor([1.0, -1.0])))
    assert torch.isfinite(weight.grad).all()
    assert torch.isfinite(bias.grad).all()


def test_linear_sigmoid_scorer_requires_parameters_when_stateless():
    scorer = cfp.LinearSigmoidScorer(num_features=1, owns_parameters=False)
    with pytest.raises(ValueError, match="weight and bias"):
        scorer(torch.ones((2, 1)))


def test_linear_sigmoid_scorer_initializes_identity_with_owned_parameters():
    scorer = cfp.LinearSigmoidScorer(num_features=2, score_sharpness=2.0)
    features = torch.tensor([[0.0, 1.0], [2.0, -3.0]], dtype=torch.float32)

    scorer.init_identity(score_sharpness=2.0, p0=0.8)
    scores = scorer(features)

    assert torch.allclose(scorer.weight, torch.zeros(2))
    assert torch.allclose(scores, torch.full((2,), 0.8))


def test_linear_sigmoid_scorer_initializes_identity_with_external_parameters():
    scorer = cfp.LinearSigmoidScorer(num_features=2, owns_parameters=False)
    weight = torch.ones(2, dtype=torch.float32)
    bias = torch.zeros(1, dtype=torch.float32)
    features = torch.tensor([[0.0, 1.0], [2.0, -3.0]], dtype=torch.float32)

    scorer.init_identity(score_sharpness=2.0, p0=0.8, weight=weight, bias=bias)
    scores = scorer(features, weight=weight, bias=bias, score_sharpness=2.0)

    assert torch.allclose(weight, torch.zeros(2))
    assert torch.allclose(scores, torch.full((2,), 0.8))


def test_mlp_scorer_forward_backpropagates():
    scorer = cfp.MLPScorer(num_features=2, hidden_units=(3,), activation="tanh", score_sharpness=1.5)
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )

    scores = scorer(features)
    scores.sum().backward()

    assert scores.shape == (3,)
    assert torch.all((scores >= 0.0) & (scores <= 1.0))
    assert scorer.to_config() == {
        "kind": "mlp",
        "hidden_units": [3],
        "activation": "tanh",
    }
    gradients = [parameter.grad for parameter in scorer.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_mlp_scorer_initializes_close_to_identity_without_cutting_gradients():
    torch.manual_seed(0)
    scorer = cfp.MLPScorer(num_features=2, hidden_units=(3,), activation="tanh", score_sharpness=1.5)
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )

    scorer.init_identity(score_sharpness=1.5, p0=0.9)
    scores = scorer(features)
    scores.sum().backward()

    assert torch.allclose(scores, torch.full((3,), 0.9), atol=1e-3)
    gradients = [parameter.grad for parameter in scorer.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


def test_preserve_root_constraint_sets_alive_root_score_to_one():
    constraint = cfp.PreserveRootConstraint()
    scores = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)
    tree_info = {
        "parent": torch.tensor([0, 0, 1], dtype=torch.long),
        "tpre": torch.tensor([0, 1, 2], dtype=torch.long),
        "tpost": torch.tensor([3, 3, 3], dtype=torch.long),
    }

    constrained = constraint(scores, tree_info)

    assert torch.allclose(constrained, torch.tensor([1.0, 0.5, 0.75]))


def test_edge_score_monotonicity_regularizer_penalizes_child_above_parent():
    regularizer = cfp.EdgeScoreMonotonicityRegularizer(weight=2.0)
    scores = torch.tensor([0.5, 0.7, 0.4], dtype=torch.float32, requires_grad=True)
    tree_info = {
        "parent": torch.tensor([0, 0, 0], dtype=torch.long),
        "tpre": torch.tensor([0, 1, 2], dtype=torch.long),
        "tpost": torch.tensor([3, 2, 3], dtype=torch.long),
    }

    penalty = regularizer(scores, tree_info)
    penalty.backward()

    assert penalty.item() == pytest.approx(0.04)
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("weight", [True, -1.0, float("nan"), float("inf"), "1"])
def test_edge_score_monotonicity_regularizer_rejects_invalid_weight(weight):
    with pytest.raises((TypeError, ValueError), match="weight"):
        cfp.EdgeScoreMonotonicityRegularizer(weight=weight)


def test_edge_score_monotonicity_regularizer_rejects_invalid_tree_shapes():
    regularizer = cfp.EdgeScoreMonotonicityRegularizer()
    tree_info = {
        "parent": torch.tensor([0, 0, 1], dtype=torch.long),
        "tpre": torch.tensor([0, 1, 2], dtype=torch.long),
        "tpost": torch.tensor([3, 3, 3], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="scores"):
        regularizer(torch.ones(1, 3), tree_info)

    with pytest.raises(ValueError, match="same number of nodes"):
        regularizer(torch.ones(2), tree_info)

    with pytest.raises(ValueError, match="tree_info must contain"):
        regularizer(torch.ones(3), {"parent": torch.tensor([0, 0, 1], dtype=torch.long)})

    with pytest.raises(ValueError, match="parent, tpre, tpost, and scores"):
        regularizer(
            torch.ones(3),
            {
                "parent": torch.tensor([0, 0, 1], dtype=torch.long),
                "tpre": torch.tensor([0, 1], dtype=torch.long),
                "tpost": torch.tensor([3, 2, 3], dtype=torch.long),
            },
        )


def test_attribute_order_score_monotonicity_regularizer_penalizes_score_inversions():
    regularizer = cfp.AttributeOrderScoreMonotonicityRegularizer(weight=2.0)
    scores = torch.tensor([0.2, 0.7, 0.4, 0.9], dtype=torch.float32, requires_grad=True)
    features = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32)
    tree_info = {
        "parent": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "tpre": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "tpost": torch.tensor([4, 4, 3, 4], dtype=torch.long),
    }

    penalty = regularizer(scores, tree_info, features=features)
    penalty.backward()

    assert penalty.item() == pytest.approx(0.06)
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0.0


def test_attribute_order_score_monotonicity_regularizer_supports_decreasing_direction():
    regularizer = cfp.AttributeOrderScoreMonotonicityRegularizer(
        weight=3.0,
        direction="decreasing",
    )
    scores = torch.tensor([0.9, 0.7, 0.8], dtype=torch.float32, requires_grad=True)
    features = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    tree_info = {
        "parent": torch.tensor([0, 0, 1], dtype=torch.long),
        "tpre": torch.tensor([0, 1, 2], dtype=torch.long),
        "tpost": torch.tensor([3, 3, 3], dtype=torch.long),
    }

    penalty = regularizer(scores, tree_info, features=features)

    assert penalty.item() == pytest.approx(0.015)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": True},
        {"weight": -1.0},
        {"feature_index": True},
        {"feature_index": -1},
        {"direction": "sideways"},
        {"min_gap": -1.0},
    ],
)
def test_attribute_order_score_monotonicity_regularizer_rejects_invalid_options(kwargs):
    with pytest.raises((TypeError, ValueError), match="weight|feature_index|direction|min_gap"):
        cfp.AttributeOrderScoreMonotonicityRegularizer(**kwargs)


def test_path_score_monotonicity_regularizer_penalizes_descendant_above_ancestor():
    regularizer = cfp.PathScoreMonotonicityRegularizer(weight=2.0, max_depth=2)
    scores = torch.tensor([0.5, 0.4, 0.8, 0.3], dtype=torch.float32, requires_grad=True)
    tree_info = {
        "parent": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "tpre": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "tpost": torch.tensor([4, 4, 3, 4], dtype=torch.long),
    }

    penalty = regularizer(scores, tree_info)
    penalty.backward()

    assert penalty.item() == pytest.approx(0.1)
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0.0


def test_path_score_monotonicity_regularizer_checks_all_ancestors_by_default():
    regularizer = cfp.PathScoreMonotonicityRegularizer()
    scores = torch.tensor([0.0, 0.8, 0.9], dtype=torch.float32, requires_grad=True)
    tree_info = {
        "parent": torch.tensor([0, 0, 1], dtype=torch.long),
        "tpre": torch.tensor([0, 1, 2], dtype=torch.long),
        "tpost": torch.tensor([3, 3, 3], dtype=torch.long),
    }

    penalty = regularizer(scores, tree_info)

    expected = (0.8**2 + 0.1**2 + 0.9**2) / 3.0
    assert penalty.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": True},
        {"weight": -1.0},
        {"max_depth": True},
        {"max_depth": 0},
    ],
)
def test_path_score_monotonicity_regularizer_rejects_invalid_options(kwargs):
    with pytest.raises((TypeError, ValueError), match="weight|max_depth"):
        cfp.PathScoreMonotonicityRegularizer(**kwargs)


@pytest.mark.parametrize(
    "regularizer, kwargs",
    [
        (cfp.EdgeScoreMonotonicityRegularizer(), {}),
        (cfp.PathScoreMonotonicityRegularizer(), {}),
        (cfp.AttributeOrderScoreMonotonicityRegularizer(), {"features": torch.ones(3, 1)}),
    ],
)
def test_regularizers_require_complete_tree_info(regularizer, kwargs):
    scores = torch.ones(3)

    with pytest.raises(ValueError, match="tree_info must contain parent, tpre, and tpost"):
        regularizer(scores, {"parent": torch.tensor([0, 0, 1], dtype=torch.long)}, **kwargs)


def test_tree_payload_cache_stores_payloads_by_sample_and_tree_key():
    cache = TreePayloadCache()
    payload = {"info": object()}

    cache.set("sample-0", "max-tree|None|0|0", payload)

    assert cache.get("sample-0", "max-tree|None|0|0") is payload
    assert cache.get("sample-1", "max-tree|None|0|0") is None
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0


def test_batch_input_normalizer_handles_plain_list_and_cached_forms():
    tensor = torch.arange(8, dtype=torch.float32).view(2, 1, 2, 2)
    idx = torch.tensor([10, 11], dtype=torch.long)
    list_batch = [tensor[0], tensor[1]]

    plain = BatchInputNormalizer.normalize(tensor)
    stacked = BatchInputNormalizer.normalize(list_batch)
    cached_tuple = BatchInputNormalizer.normalize((tensor, idx))
    cached_list = BatchInputNormalizer.normalize([tensor, idx])

    assert plain.tensor is tensor
    assert plain.index.tolist() == [0, 1]
    assert plain.use_cache is False
    assert torch.equal(stacked.tensor, tensor)
    assert stacked.index.tolist() == [0, 1]
    assert stacked.use_cache is False
    assert cached_tuple.tensor is tensor
    assert cached_tuple.index is idx
    assert cached_tuple.use_cache is True
    assert cached_list.tensor is tensor
    assert cached_list.index is idx
    assert cached_list.use_cache is True


def test_cached_dataloader_builder_wraps_dataset_with_stable_indices():
    dataset = TensorDataset(
        torch.arange(4, dtype=torch.float32).view(4, 1),
        torch.arange(4),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    wrapped = CachedDataLoaderBuilder.wrap_dataloader(loader, index_offset=50)
    (x, idx), y = next(iter(wrapped))

    assert wrapped.batch_size == 2
    assert x.shape == (2, 1)
    assert idx.tolist() == [50, 51]
    assert y.tolist() == [0, 1]


def test_stats_serializer_roundtrips_tensors_to_requested_device():
    stats = {
        "max-tree|None|0|0::AREA": {
            "count": torch.tensor(3.0),
            "sum": torch.tensor(6.0),
            "sumsq": torch.tensor(14.0),
            "label": "area",
        }
    }
    serializer = StatsSerializer()

    payload = serializer.serialize(stats)
    restored = serializer.deserialize(payload, device=torch.device("cpu"))

    assert payload["max-tree|None|0|0::AREA"]["sum"].device.type == "cpu"
    assert restored["max-tree|None|0|0::AREA"]["sum"].device.type == "cpu"
    assert torch.equal(restored["max-tree|None|0|0::AREA"]["sumsq"], torch.tensor(14.0))
    assert restored["max-tree|None|0|0::AREA"]["label"] == "area"


def test_config_deserializer_resolves_serialized_layer_kwargs():
    deserializer = ConfigDeserializer()
    kwargs = deserializer.deserialize_layer_config(
        {
            "in_channels": 1,
            "filter_specs": [
                {
                    "name": "tos_mean",
                    "tree_type": "tree-of-shapes",
                    "attributes": ["BOUNDARY"],
                    "tos_interpolation": "Min8cMax4c",
                    "regularizers": [{"kind": "edge_score_monotonicity", "weight": 0.25}],
                }
            ],
            "scale_mode": "none",
        }
    )

    spec = kwargs["filter_specs"][0]

    assert kwargs["in_channels"] == 1
    assert kwargs["scale_mode"] == "none"
    assert spec["name"] == "tos_mean"
    assert spec["attributes"] == (morphology.AttributeGroup.BOUNDARY,)
    assert spec["tos_interpolation"] == morphology.ToSInterpolation.Min8cMax4c
    assert spec["regularizers"] == [{"kind": "edge_score_monotonicity", "weight": 0.25}]


def test_filter_spec_collects_scoring_attributes_once():
    tree = cfp.TreeSpec(morphology.TreeType.MAX_TREE)
    features = cfp.FeatureSpec((morphology.AttributeType.AREA,))
    spec = cfp.FilterSpec(
        name="area",
        tree=tree,
        features=features,
        scoring=object(),
    )

    assert tree.cache_key() == f"{morphology.TreeType.MAX_TREE}|None|0|0"
    assert spec.all_required_attributes() == (morphology.AttributeType.AREA,)


def test_filter_spec_normalizer_builds_internal_spec_contract():
    normalized = normalize_filter_specs(
        [
            {
                "name": "tos_boundary",
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": morphology.AttributeGroup.BOUNDARY,
                "scoring": {"kind": "linear_sigmoid"},
                "constraints": ["preserve_root"],
                "regularizers": [{"kind": "edge_score_monotonicity", "weight": 0.25}],
                "tos_interpolation": morphology.ToSInterpolation.Min8cMax4c,
                "tos_infinity_seed_row": 1,
                "tos_infinity_seed_col": 2,
            }
        ],
        default_tos_interpolation=None,
        default_tos_infinity_seed_row=0,
        default_tos_infinity_seed_col=0,
    )

    spec = normalized[0]

    assert spec.index == 0
    assert spec.key == "tos_boundary"
    assert spec.tree_type == "tree-of-shapes"
    assert spec.tree_key == "tree-of-shapes|Min8cMax4c|1|2"
    assert morphology.AttributeType.MAX_DIST not in spec.attributes
    assert isinstance(spec.scoring_model, cfp.LinearSigmoidScorer)
    assert spec.constraint_configs == ({"kind": "preserve_root"},)
    assert spec.regularizer_configs == ({"kind": "edge_score_monotonicity", "weight": 0.25},)


def test_layer_owned_linear_parameter_initializer_only_creates_layer_owned_parameters():
    from mtlearn.layers.cfp.scoring._layer_owned_linear_parameter_initializer import (
        LayerOwnedLinearParameterInitializer,
    )

    specs = normalize_filter_specs(
        [
            {
                "name": "linear_area",
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            },
                {
                    "name": "mlp_area",
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                    "scoring": {"kind": "mlp", "hidden_units": [2]},
                },
            ],
        default_tos_interpolation=None,
        default_tos_infinity_seed_row=0,
        default_tos_infinity_seed_col=0,
    )
    initializer = LayerOwnedLinearParameterInitializer()

    weights, biases = initializer.create_parameter_dicts(specs, device="cpu")
    initializer.init_identity_with_bias(
        specs,
        weights,
        biases,
        score_sharpness=2.0,
        p0=0.8,
    )
    expected_bias = LayerOwnedLinearParameterInitializer.logit(0.8) / 2.0

    assert set(weights) == {"linear_area"}
    assert set(biases) == {"linear_area"}
    assert weights["linear_area"].shape == torch.Size([1])
    assert biases["linear_area"].shape == torch.Size([1])
    assert torch.allclose(weights["linear_area"], torch.zeros(1))
    assert torch.allclose(biases["linear_area"], torch.tensor([expected_bias]))


def test_spec_registry_creates_components_with_context():
    registry = cfp.SpecRegistry()

    def make_component(*, num_features, gain=1):
        return {"num_features": num_features, "gain": gain}

    registry.register("demo", make_component)
    serializer = ConfigSerializer(registry)

    component = serializer.from_config(
        {"kind": "demo", "gain": 2},
        num_features=3,
    )

    assert component == {"num_features": 3, "gain": 2}
    assert registry.registered_kinds() == ("demo",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("demo", make_component)


def test_default_component_registries_expose_extension_kinds():
    scorer = cfp.SCORING_MODEL_REGISTRY.create({"kind": "linear_sigmoid"}, num_features=2)
    constraint = cfp.SCORE_CONSTRAINT_REGISTRY.create({"kind": "preserve_root"})
    regularizer = cfp.REGULARIZER_REGISTRY.create(
        {
            "kind": "edge_score_monotonicity",
            "weight": 0.5,
        }
    )
    attribute_regularizer = cfp.REGULARIZER_REGISTRY.create(
        {
            "kind": "attribute_order_score_monotonicity",
            "weight": 0.5,
        }
    )
    path_regularizer = cfp.REGULARIZER_REGISTRY.create(
        {
            "kind": "path_score_monotonicity",
            "max_depth": 2,
        }
    )

    assert isinstance(scorer, cfp.LinearSigmoidScorer)
    assert isinstance(constraint, cfp.PreserveRootConstraint)
    assert isinstance(regularizer, cfp.EdgeScoreMonotonicityRegularizer)
    assert isinstance(attribute_regularizer, cfp.AttributeOrderScoreMonotonicityRegularizer)
    assert isinstance(path_regularizer, cfp.PathScoreMonotonicityRegularizer)
    assert "mlp" in cfp.SCORING_MODEL_REGISTRY.registered_kinds()
    with pytest.raises(KeyError, match="unknown CFP component kind"):
        cfp.SCORING_MODEL_REGISTRY.create({"kind": "linear-sigmoid"}, num_features=2)
    with pytest.raises(ValueError, match="unsupported mlp scoring options"):
        cfp.SCORING_MODEL_REGISTRY.create({"kind": "mlp", "hidden": [2]}, num_features=2)
    with pytest.raises(ValueError, match="unsupported mlp scoring options"):
        cfp.SCORING_MODEL_REGISTRY.create({"kind": "mlp", "hidden_channels": [2]}, num_features=2)


@pytest.mark.parametrize(
    "unknown_kind",
    [
        "monotone_scores",
        "monotone-scores",
        "attribute_order_monotonicity",
        "attribute-order-monotonicity",
        "ancestor_consistency",
        "ancestor-consistency",
    ],
)
def test_regularizer_registry_rejects_unregistered_names(unknown_kind):
    with pytest.raises(KeyError, match="unknown CFP component kind"):
        cfp.REGULARIZER_REGISTRY.create({"kind": unknown_kind})
