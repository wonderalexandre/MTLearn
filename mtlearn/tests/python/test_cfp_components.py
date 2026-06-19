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
from mtlearn.layers.cfp.filter_spec_normalizer import normalize_filter_specs

pytestmark = pytest.mark.integration


def test_grouped_cfp_subpackages_preserve_public_component_identity():
    from mtlearn.layers.cfp.constraints import PreserveRootConstraint
    from mtlearn.layers.cfp.mlp_scorer import MLPScorer as LegacyMLPScorer
    from mtlearn.layers.cfp.normalization import AttributeNormalizer
    from mtlearn.layers.cfp.regularization import MonotoneScoresRegularizer
    from mtlearn.layers.cfp.runtime import TreeReconstructor
    from mtlearn.layers.cfp.scoring import MLPScorer
    from mtlearn.layers.cfp.serialization import ConfigSerializer
    from mtlearn.layers.cfp.specs import FilterSpec
    from mtlearn.layers.cfp.valuation import AltitudeValuation

    assert MLPScorer is cfp.MLPScorer
    assert LegacyMLPScorer is cfp.MLPScorer
    assert AltitudeValuation is cfp.AltitudeValuation
    assert PreserveRootConstraint is cfp.PreserveRootConstraint
    assert MonotoneScoresRegularizer is cfp.MonotoneScoresRegularizer
    assert AttributeNormalizer is cfp.AttributeNormalizer
    assert TreeReconstructor is cfp.TreeReconstructor
    assert ConfigSerializer is cfp.ConfigSerializer
    assert FilterSpec is cfp.FilterSpec


def test_linear_sigmoid_scorer_matches_explicit_formula():
    scorer = cfp.LinearSigmoidScorer(num_features=2, beta_f=2.0)
    with torch.no_grad():
        scorer.weight.copy_(torch.tensor([1.0, -1.0]))
        scorer.bias.zero_()
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

    scores = scorer(features)

    expected = torch.sigmoid(torch.tensor([2.0, -2.0, 0.0], dtype=torch.float32))
    assert torch.allclose(scores, expected)


def test_linear_sigmoid_scorer_accepts_external_parameters():
    scorer = cfp.LinearSigmoidScorer(num_features=2, beta_f=1.0, owns_parameters=False)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    weight = torch.tensor([2.0, -2.0], dtype=torch.float32, requires_grad=True)
    bias = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

    scores = scorer(features, weight=weight, bias=bias, beta_f=0.5)
    scores.sum().backward()

    assert torch.allclose(scores, torch.sigmoid(torch.tensor([1.0, -1.0])))
    assert torch.isfinite(weight.grad).all()
    assert torch.isfinite(bias.grad).all()


def test_linear_sigmoid_scorer_requires_parameters_when_stateless():
    scorer = cfp.LinearSigmoidScorer(num_features=1, owns_parameters=False)
    with pytest.raises(ValueError, match="weight and bias"):
        scorer(torch.ones((2, 1)))


def test_mlp_scorer_forward_backpropagates():
    scorer = cfp.MLPScorer(num_features=2, hidden_channels=(3,), activation="tanh", beta_f=1.5)
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
        "hidden_channels": [3],
        "activation": "tanh",
    }
    gradients = [parameter.grad for parameter in scorer.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


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


def test_monotone_scores_regularizer_penalizes_child_above_parent():
    regularizer = cfp.MonotoneScoresRegularizer(weight=2.0)
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
def test_monotone_scores_regularizer_rejects_invalid_weight(weight):
    with pytest.raises((TypeError, ValueError), match="weight"):
        cfp.MonotoneScoresRegularizer(weight=weight)


def test_tree_payload_cache_stores_payloads_by_sample_and_tree_key():
    cache = cfp.TreePayloadCache()
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

    plain = cfp.BatchInputNormalizer.normalize(tensor)
    stacked = cfp.BatchInputNormalizer.normalize(list_batch)
    cached_tuple = cfp.BatchInputNormalizer.normalize((tensor, idx))
    cached_list = cfp.BatchInputNormalizer.normalize([tensor, idx])

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

    wrapped = cfp.CachedDataLoaderBuilder.wrap_dataloader(loader, index_offset=50)
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
    serializer = cfp.StatsSerializer()

    payload = serializer.serialize(stats)
    restored = serializer.deserialize(payload, device=torch.device("cpu"))

    assert payload["max-tree|None|0|0::AREA"]["sum"].device.type == "cpu"
    assert restored["max-tree|None|0|0::AREA"]["sum"].device.type == "cpu"
    assert torch.equal(restored["max-tree|None|0|0::AREA"]["sumsq"], torch.tensor(14.0))
    assert restored["max-tree|None|0|0::AREA"]["label"] == "area"


def test_config_deserializer_resolves_serialized_layer_kwargs():
    deserializer = cfp.ConfigDeserializer()
    kwargs = deserializer.deserialize_layer_config(
        {
            "in_channels": 1,
            "filter_specs": [
                {
                    "name": "tos_mean",
                    "tree_type": "tree-of-shapes",
                    "attributes": ["BOUNDARY"],
                    "valuation": {
                        "kind": "node_attribute",
                        "attribute": "MEAN_LEVEL",
                    },
                    "tos_interpolation": "Min8cMax4c",
                    "monotonicity_weight": 0.25,
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
    assert spec["valuation"] == cfp.CFPValuation.node_attribute(morphology.AttributeType.MEAN_LEVEL)
    assert spec["tos_interpolation"] == morphology.ToSInterpolation.Min8cMax4c
    assert spec["monotonicity_weight"] == pytest.approx(0.25)


def test_filter_spec_collects_scoring_and_valuation_attributes_once():
    tree = cfp.TreeSpec(morphology.TreeType.MAX_TREE)
    features = cfp.FeatureSpec((morphology.AttributeType.AREA,))
    valuation = cfp.NodeAttributeValuation(morphology.AttributeType.AREA)
    spec = cfp.FilterSpec(
        name="area",
        tree=tree,
        features=features,
        scoring=object(),
        valuation=valuation,
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
                "valuation": cfp.CFPValuation.ALTITUDE_TOPHAT,
                "constraints": ["preserve_root"],
                "regularizers": [{"kind": "monotone_scores", "weight": 0.25}],
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
    assert spec.valuation == cfp.CFPValuation.ALTITUDE_TOPHAT
    assert isinstance(spec.valuation_projection, cfp.AltitudeTopHatValuation)
    assert isinstance(spec.scoring_model, cfp.LinearSigmoidScorer)
    assert spec.preserve_root is True
    assert spec.constraint_configs == ({"kind": "preserve_root"},)
    assert spec.regularizer_configs == ({"kind": "monotone_scores", "weight": 0.25},)


def test_legacy_linear_parameter_initializer_only_creates_legacy_parameters():
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
                "scoring": {"kind": "mlp", "hidden_channels": [2]},
            },
        ],
        default_tos_interpolation=None,
        default_tos_infinity_seed_row=0,
        default_tos_infinity_seed_col=0,
    )
    initializer = cfp.LegacyLinearParameterInitializer()

    weights, biases = initializer.create_parameter_dicts(specs, device="cpu")
    initializer.init_identity_with_bias(
        specs,
        weights,
        biases,
        beta_f=2.0,
        p0=0.8,
    )
    expected_bias = cfp.LegacyLinearParameterInitializer.logit(0.8) / 2.0

    assert set(weights) == {"linear_area"}
    assert set(biases) == {"linear_area"}
    assert weights["linear_area"].shape == torch.Size([1])
    assert biases["linear_area"].shape == torch.Size([1])
    assert torch.allclose(weights["linear_area"], torch.zeros(1))
    assert torch.allclose(biases["linear_area"], torch.tensor([expected_bias]))

    initializer.init_identity_bias_zero(
        specs,
        weights,
        biases,
        beta_f=2.0,
        hybrid_floor_a=0.5,
        p0=0.8,
    )

    assert torch.allclose(weights["linear_area"], torch.tensor([cfp.LegacyLinearParameterInitializer.logit(0.8)]))
    assert torch.allclose(biases["linear_area"], torch.zeros(1))


def test_spec_registry_creates_components_with_aliases_and_context():
    registry = cfp.SpecRegistry()

    def make_component(*, num_features, gain=1):
        return {"num_features": num_features, "gain": gain}

    registry.register("demo", make_component, aliases=("demo-alias",))
    serializer = cfp.ConfigSerializer(registry)

    component = serializer.from_config(
        {"kind": "demo-alias", "gain": 2},
        num_features=3,
    )

    assert component == {"num_features": 3, "gain": 2}
    assert registry.registered_kinds() == ("demo", "demo-alias")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("demo", make_component)


def test_default_component_registries_expose_extension_kinds():
    scorer = cfp.SCORING_MODEL_REGISTRY.create({"kind": "linear-sigmoid"}, num_features=2)
    projection = cfp.VALUATION_PROJECTION_REGISTRY.create(
        {
            "kind": "node_attribute",
            "attribute": morphology.AttributeType.AREA,
        }
    )
    constraint = cfp.SCORE_CONSTRAINT_REGISTRY.create({"kind": "preserve_root"})
    regularizer = cfp.REGULARIZER_REGISTRY.create(
        {
            "kind": "monotone-scores",
            "weight": 0.5,
        }
    )

    assert isinstance(scorer, cfp.LinearSigmoidScorer)
    assert isinstance(projection, cfp.NodeAttributeValuation)
    assert isinstance(constraint, cfp.PreserveRootConstraint)
    assert isinstance(regularizer, cfp.MonotoneScoresRegularizer)
    assert "mlp" in cfp.SCORING_MODEL_REGISTRY.registered_kinds()
    assert "altitude_tophat" in cfp.VALUATION_PROJECTION_REGISTRY.registered_kinds()


def test_valuation_projection_keys_and_tophat_projection():
    altitude = cfp.AltitudeValuation()
    tophat = cfp.AltitudeTopHatValuation()
    node_attribute = cfp.NodeAttributeValuation(morphology.AttributeType.AREA)
    filtered = torch.tensor([[1.0, 2.0]])
    unfiltered = torch.tensor([[3.0, 1.0]])

    assert altitude.key() == "altitude"
    assert tophat.key() == "altitude_tophat"
    assert node_attribute.key() == "node_attribute:AREA"
    assert node_attribute.required_attributes() == (morphology.AttributeType.AREA,)
    assert altitude.project(filtered, None, {}) is filtered
    assert tophat.requires_unfiltered_image() is True
    assert torch.allclose(
        tophat.project(filtered, unfiltered, {"tree_type": morphology.TreeType.MAX_TREE.value}),
        unfiltered - filtered,
    )
