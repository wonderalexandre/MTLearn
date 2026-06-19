import pytest

import mtlearn

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import torch
except Exception as exc:  # pragma: no cover
    pytest.skip(f"PyTorch unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology
from mtlearn.layers import cfp
from mtlearn.layers import (
    CFPValuation,
    ConnectedFilterPreprocessingLayer,
    ConnectedFilterPreprocessingLayerLegacy,
)
from mtlearn.layers._helpers import IndexedDatasetWrapper, deserialize_ds_stats

pytestmark = pytest.mark.integration


def _single_area_layer():
    return ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            }
        ],
        device="cpu",
        scale_mode="none",
    )


def test_constructor_rejects_empty_attribute_group():
    with pytest.raises(ValueError, match="at least one attribute"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[{"tree_type": morphology.TreeType.MAX_TREE, "attributes": ()}],
            device="cpu",
        )


def test_constructor_normalizes_clamp_parameter():
    scalar_layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            }
        ],
        device="cpu",
        scale_mode="none",
        clamp=12,
    )
    pair_layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            }
        ],
        device="cpu",
        scale_mode="none",
        clamp=(-8, 10),
    )

    assert scalar_layer.clamp == (-12.0, 12.0)
    assert pair_layer.clamp == (-8.0, 10.0)


@pytest.mark.parametrize(
    "clamp",
    [True, False, 0, -1, float("nan"), (1, 1), (2, -2), (1, 2, 3)],
)
def test_constructor_rejects_invalid_clamp_parameter(clamp):
    with pytest.raises((TypeError, ValueError)):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                }
            ],
            device="cpu",
            scale_mode="none",
            clamp=clamp,
        )


@pytest.mark.parametrize(
    ("attribute_dtype", "expected_name"),
    [
        (torch.float32, "float32"),
        (torch.float64, "float64"),
        ("float64", "float64"),
    ],
)
def test_constructor_normalizes_attribute_dtype(attribute_dtype, expected_name):
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            }
        ],
        device="cpu",
        scale_mode="none",
        attribute_dtype=attribute_dtype,
    )
    restored = ConnectedFilterPreprocessingLayer.from_config(layer.get_config())

    assert layer.attribute_dtype.name == expected_name
    assert layer.get_config()["attribute_dtype"] == expected_name
    assert restored.attribute_dtype.name == expected_name


def test_constructor_rejects_invalid_attribute_dtype():
    with pytest.raises((TypeError, ValueError), match="attribute_dtype"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                }
            ],
            device="cpu",
            scale_mode="none",
            attribute_dtype=torch.float16,
        )


def test_attribute_dtype_controls_cached_attributes_and_valuations():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
                "valuation": CFPValuation.node_attribute(morphology.AttributeType.AREA),
            }
        ],
        device="cpu",
        scale_mode="none",
        attribute_dtype=torch.float64,
    )
    image = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    inspected = layer.inspect_training_sample(image)["specs"]["spec_000"]
    output = layer(image.reshape(1, 1, 2, 2))

    assert inspected["base_attrs"].dtype == torch.float64
    assert inspected["norm_attrs"].dtype == torch.float64
    assert inspected["valuation_increments"].dtype == torch.float64
    assert output.dtype == torch.float32


@pytest.mark.parametrize("name", ["", "has.dot", "1starts_with_digit", "with space", object()])
def test_constructor_rejects_invalid_filter_spec_name(name):
    with pytest.raises((TypeError, ValueError), match="filter spec name"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "name": name,
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                }
            ],
            device="cpu",
            scale_mode="none",
        )


def test_constructor_rejects_duplicate_filter_spec_names():
    with pytest.raises(ValueError, match="duplicate filter spec name"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "name": "area",
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                },
                {
                    "name": "area",
                    "tree_type": morphology.TreeType.MIN_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                },
            ],
            device="cpu",
            scale_mode="none",
        )


def test_constructor_rejects_removed_clamp_logits_parameter():
    with pytest.raises(TypeError, match="clamp_logits"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                }
            ],
            device="cpu",
            scale_mode="none",
            clamp_logits=False,
        )


@pytest.mark.parametrize(
    "attributes_spec",
    [
        [(morphology.AttributeType.MAX_DIST,)],
        [(morphology.AttributeGroup.SHAPE, morphology.AttributeType.MAX_DIST)],
    ],
)
def test_tree_of_shapes_constructor_rejects_explicit_unsupported_attributes(attributes_spec):
    with pytest.raises(ValueError, match="tree-of-shapes CFP does not support"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                    "attributes": attributes_spec[0],
                }
            ],
            device="cpu",
        )


@pytest.mark.parametrize(
    "group",
    [
        morphology.AttributeGroup.SHAPE,
        morphology.AttributeGroup.ALL,
    ],
)
def test_tree_of_shapes_constructor_omits_max_dist_from_supported_groups(group):
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": (group,),
            }
        ],
        device="cpu",
        scale_mode="none",
    )
    expected_group = tuple(
        attr
        for attr in morphology.expand_attribute_group(group)
        if attr != morphology.AttributeType.MAX_DIST
    )

    assert layer.filter_specs[0].tree_type == "tree-of-shapes"
    assert layer.filter_specs[0].attributes == expected_group
    assert morphology.AttributeType.MAX_DIST not in layer.filter_specs[0].attributes


def test_tree_of_shapes_constructor_accepts_boundary_group():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": (morphology.AttributeGroup.BOUNDARY,),
            }
        ],
        device="cpu",
        scale_mode="none",
    )

    assert layer.filter_specs[0].tree_type == "tree-of-shapes"
    assert len(layer.filter_specs) == 1
    assert tuple(morphology.expand_attribute_group(morphology.AttributeGroup.BOUNDARY)) == layer.filter_specs[0].attributes
    assert morphology.AttributeType.MAX_DIST not in layer.filter_specs[0].attributes


@pytest.mark.parametrize(
    "group",
    [
        morphology.AttributeGroup.SHAPE,
        morphology.AttributeGroup.ALL,
    ],
)
def test_tree_of_shapes_forward_accepts_filtered_attribute_groups(group):
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": (group,),
            }
        ],
        device="cpu",
        scale_mode="none",
    )
    image = torch.tensor(
        [[[[0.0, 0.2, 0.4], [0.1, 0.8, 0.3], [0.5, 0.7, 1.0]]]],
        dtype=torch.float32,
    )

    output = layer(image)

    assert output.shape == (1, 1, 3, 3)


def test_tree_of_shapes_forward_accepts_boundary_attributes():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": (morphology.AttributeType.BITQUADS_AREA,),
            },
            {
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": (morphology.AttributeType.CONTOUR_PERIMETER,),
            },
        ],
        device="cpu",
        scale_mode="none",
    )
    image = torch.tensor(
        [[[[0.0, 0.2, 0.4], [0.1, 0.8, 0.3], [0.5, 0.7, 1.0]]]],
        dtype=torch.float32,
    )

    output = layer(image)

    assert output.shape == (1, 2, 3, 3)


def test_filter_spec_defaults_to_filtered_altitude_valuation():
    layer = _single_area_layer()

    assert layer.filter_specs[0].valuation == CFPValuation.ALTITUDE
    assert isinstance(layer.filter_specs[0].valuation_projection, cfp.AltitudeValuation)
    assert isinstance(layer.filter_specs[0].scoring_model, cfp.LinearSigmoidScorer)
    assert layer.get_config()["filter_specs"][0]["scoring"] == {"kind": "linear_sigmoid"}


def test_filter_spec_rejects_unknown_valuation_kind():
    with pytest.raises(ValueError, match="unknown CFP valuation kind"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                    "valuation": CFPValuation("unknown"),
                }
            ],
            device="cpu",
            scale_mode="none",
        )


def test_filter_spec_accepts_explicit_linear_sigmoid_scoring_config():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
                "scoring": {"kind": "linear_sigmoid"},
            }
        ],
        device="cpu",
        scale_mode="none",
    )
    restored = ConnectedFilterPreprocessingLayer.from_config(layer.get_config(), device="cpu")

    assert isinstance(layer.filter_specs[0].scoring_model, cfp.LinearSigmoidScorer)
    assert restored.get_config() == layer.get_config()


def test_filter_spec_accepts_mlp_scoring_config_and_registers_model_parameters():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "name": "mlp_area",
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (
                    morphology.AttributeType.AREA,
                    morphology.AttributeType.COMPACTNESS,
                ),
                "scoring": {
                    "kind": "mlp",
                    "hidden_channels": [4],
                    "activation": "tanh",
                },
            }
        ],
        device="cpu",
        scale_mode="none",
    )
    image = torch.tensor(
        [[[[0.0, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.7, 1.0]]]],
        dtype=torch.float32,
    )

    output = layer(image)
    output.mean().backward()
    restored = ConnectedFilterPreprocessingLayer.from_config(layer.get_config(), device="cpu")
    state_keys = set(layer.state_dict())
    parameter_contract = layer.get_parameter_contract()

    assert output.shape == (1, 1, 3, 3)
    assert isinstance(layer.filter_specs[0].scoring_model, cfp.MLPScorer)
    assert layer.get_config()["filter_specs"][0]["scoring"] == {
        "kind": "mlp",
        "hidden_channels": [4],
        "activation": "tanh",
    }
    assert restored.get_config() == layer.get_config()
    assert set(layer._weights) == set()
    assert set(layer._biases) == set()
    assert "_weights.mlp_area" not in state_keys
    assert "_biases.mlp_area" not in state_keys
    assert any(key.startswith("_scoring_models.mlp_area.network") for key in state_keys)
    assert parameter_contract["weights"] == {}
    assert parameter_contract["biases"] == {}
    assert "mlp_area" in parameter_contract["scoring_models"]
    gradients = [parameter.grad for parameter in layer._scoring_models["mlp_area"].parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_custom_scoring_model_receives_cfp_context():
    class RecordingScorer(cfp.ScoringModel):
        def __init__(self):
            super().__init__()
            self.num_features = 1
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.contexts = []

        def forward(self, features, tree_info=None, context=None, **kwargs):
            self.contexts.append(context)
            return torch.sigmoid(features[:, 0] * 0.0 + self.bias)

        def to_config(self):
            return {"kind": "recording"}

    scorer = RecordingScorer()
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "name": "context_area",
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
                "scoring": scorer,
            }
        ],
        device="cpu",
        scale_mode="none",
    )
    image = torch.tensor(
        [[[[0.0, 0.2], [0.8, 1.0]]]],
        dtype=torch.float32,
    )

    layer(image)

    assert len(scorer.contexts) == 1
    context = scorer.contexts[0]
    assert isinstance(context, cfp.CFPContext)
    assert context.sample_key == "0_0"
    assert context.batch_index == 0
    assert context.channel_index == 0
    assert context.spec_name == "context_area"
    assert context.extras["mode"] == "forward"


def test_filter_spec_rejects_unknown_scoring_config():
    with pytest.raises(ValueError, match="scoring model"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                    "scoring": {"kind": "unknown"},
                }
            ],
            device="cpu",
            scale_mode="none",
        )


def test_filter_spec_accepts_declarative_constraints_and_regularizers():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "name": "regularized_area",
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
                "constraints": [{"kind": "preserve_root"}],
                "regularizers": [{"kind": "monotone_scores", "weight": 2.0}],
            }
        ],
        device="cpu",
        scale_mode="none",
    )
    with torch.no_grad():
        layer._weights["regularized_area"].fill_(-4.0)
        layer._biases["regularized_area"].zero_()
    image = torch.tensor(
        [[[[0.0, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.7, 1.0]]]],
        dtype=torch.float32,
    )

    output = layer(image)
    penalty = layer.monotonicity_penalty(image)
    penalty.backward()
    restored = ConnectedFilterPreprocessingLayer.from_config(layer.get_config(), device="cpu")

    assert output.shape == (1, 1, 3, 3)
    assert layer.filter_specs[0].preserve_root is True
    assert layer.filter_specs[0].monotonicity_weight == 0.0
    assert len(layer._score_constraints["regularized_area"]) == 1
    assert len(layer._regularizers["regularized_area"]) == 1
    assert penalty.item() > 0.0
    assert torch.isfinite(layer._weights["regularized_area"].grad).all()
    assert layer.get_config()["filter_specs"][0]["constraints"] == [{"kind": "preserve_root"}]
    assert layer.get_config()["filter_specs"][0]["regularizers"] == [
        {"kind": "monotone_scores", "weight": 2.0}
    ]
    assert "regularizers" not in layer.get_weight_contract()["filter_specs"][0]
    assert restored.get_config() == layer.get_config()


def test_filter_spec_rejects_unknown_constraint_and_regularizer_configs():
    with pytest.raises(ValueError, match="constraint kind"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                    "constraints": [{"kind": "unknown"}],
                }
            ],
            device="cpu",
            scale_mode="none",
        )

    with pytest.raises(ValueError, match="regularizer kind"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                    "regularizers": [{"kind": "unknown"}],
                }
            ],
            device="cpu",
            scale_mode="none",
        )


def test_linear_sigmoid_scoring_preserves_legacy_parameter_names():
    layer = _single_area_layer()
    state_keys = set(layer.state_dict())

    assert "_weights.spec_000" in state_keys
    assert "_biases.spec_000" in state_keys
    assert not any(key.startswith("_scoring_models.spec_000") for key in state_keys)


def test_cfp_package_preserves_legacy_default_layer_contract(tmp_path):
    from mtlearn.layers.ConnectedFilterPreprocessingLayer import (
        CFPValuation as LegacyCFPValuation,
    )
    from mtlearn.layers.ConnectedFilterPreprocessingLayer import (
        ConnectedFilterPreprocessingImplicitJacobianFunction as LegacyImplicitFunction,
    )
    from mtlearn.layers.ConnectedFilterPreprocessingLayer import (
        ConnectedFilterPreprocessingLayer as LegacyLayer,
    )

    assert LegacyLayer is ConnectedFilterPreprocessingLayer
    assert LegacyLayer is cfp.ConnectedFilterPreprocessingLayer
    assert LegacyCFPValuation is CFPValuation
    assert LegacyCFPValuation is cfp.CFPValuation
    assert LegacyImplicitFunction is cfp.ConnectedFilterPreprocessingImplicitJacobianFunction

    reference = LegacyLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            }
        ],
        device="cpu",
        scale_mode="none",
    )
    restored = cfp.ConnectedFilterPreprocessingLayer.from_config(
        reference.get_config(),
        device="cpu",
    )
    restored.load_state_dict(reference.state_dict())
    image = torch.tensor(
        [[[[0.0, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.7, 1.0]]]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        reference_output = reference(image)
        restored_output = restored(image)

    state_keys = set(reference.state_dict())
    weight_contract = reference.get_weight_contract()
    first_filter_contract = weight_contract["filter_specs"][0]
    export_path = tmp_path / "legacy_default_params.pt"

    reference.export_params(export_path)
    payload = torch.load(export_path, map_location="cpu", weights_only=True)

    assert torch.allclose(restored_output, reference_output)
    assert "_weights.spec_000" in state_keys
    assert "_biases.spec_000" in state_keys
    assert not any(key.startswith("_scoring_models.spec_000") for key in state_keys)
    assert "regularizers" not in first_filter_contract
    assert "monotonicity_weight" not in first_filter_contract
    assert payload["weights"]["spec_000"].shape == torch.Size([1])
    assert payload["biases"]["spec_000"].shape == torch.Size([1])
    assert payload["weight_contract"] == weight_contract
    assert payload["contracts"]["inference_contract"] == reference.get_inference_contract()


def test_preserve_root_defaults_to_false_and_roundtrips_in_config():
    layer = _single_area_layer()

    restored = ConnectedFilterPreprocessingLayer.from_config(layer.get_config(), device="cpu")

    assert layer.filter_specs[0].preserve_root is False
    assert layer.get_config()["filter_specs"][0]["preserve_root"] is False
    assert restored.filter_specs[0].preserve_root is False
    assert restored.get_weight_contract() == layer.get_weight_contract()


def test_monotonicity_weight_defaults_to_zero_and_stays_out_of_weight_contract():
    layer = _single_area_layer()

    restored = ConnectedFilterPreprocessingLayer.from_config(layer.get_config(), device="cpu")

    assert layer.filter_specs[0].monotonicity_weight == 0.0
    assert layer.get_config()["filter_specs"][0]["monotonicity_weight"] == 0.0
    assert "monotonicity_weight" not in layer.get_weight_contract()["filter_specs"][0]
    assert restored.filter_specs[0].monotonicity_weight == 0.0
    assert restored.get_weight_contract() == layer.get_weight_contract()


@pytest.mark.parametrize(
    "monotonicity_weight",
    [True, False, -1.0, float("nan"), float("inf"), "1", object()],
)
def test_filter_spec_rejects_invalid_monotonicity_weight(monotonicity_weight):
    with pytest.raises((TypeError, ValueError), match="monotonicity_weight"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                    "monotonicity_weight": monotonicity_weight,
                }
            ],
            device="cpu",
            scale_mode="none",
        )


def test_monotonicity_penalty_defaults_to_zero_without_building_payload():
    layer = _single_area_layer()
    image = torch.tensor(
        [[[[0.0, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.7, 1.0]]]],
        dtype=torch.float32,
    )

    penalty = layer.monotonicity_penalty(image)

    assert penalty.item() == 0.0
    assert penalty.requires_grad
    assert layer._tree_info == {}


def test_monotonicity_penalty_is_positive_and_backpropagates_when_enabled():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
                "monotonicity_weight": 2.0,
            }
        ],
        device="cpu",
        scale_mode="none",
    )
    with torch.no_grad():
        layer._weights["spec_000"].fill_(-4.0)
        layer._biases["spec_000"].zero_()
    image = torch.tensor(
        [[[[0.0, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.7, 1.0]]]],
        dtype=torch.float32,
    )

    penalty = layer.monotonicity_penalty(image)
    penalty.backward()

    assert penalty.item() > 0.0
    assert torch.isfinite(layer._weights["spec_000"].grad).all()
    assert torch.isfinite(layer._biases["spec_000"].grad).all()
    assert layer._weights["spec_000"].grad.abs().sum().item() > 0.0


def test_preserve_root_keeps_constant_image_when_enabled():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
                "preserve_root": True,
            }
        ],
        device="cpu",
        scale_mode="none",
        beta_f=1.0,
        clamp=None,
    )
    with torch.no_grad():
        layer._weights["spec_000"].zero_()
        layer._biases["spec_000"].zero_()
    image = torch.full((1, 1, 2, 2), 7.0, dtype=torch.float32)

    output = layer(image)

    assert layer.filter_specs[0].preserve_root is True
    assert isinstance(layer.filter_specs[0].valuation_projection, cfp.AltitudeValuation)
    assert layer.get_config()["filter_specs"][0]["preserve_root"] is True
    assert torch.allclose(output, image)


def test_filter_specs_can_mix_tree_types_and_valuations():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            },
            {
                "tree_type": morphology.TreeType.MIN_TREE,
                "attributes": (morphology.AttributeType.AREA,),
                "valuation": CFPValuation.ALTITUDE_TOPHAT,
            },
            {
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": (morphology.AttributeGroup.BOUNDARY,),
                "valuation": CFPValuation.node_attribute(morphology.AttributeType.MEAN_LEVEL),
            },
        ],
        device="cpu",
        scale_mode="none",
    )
    image = torch.tensor(
        [[[[0.0, 0.2, 0.4], [0.1, 0.8, 0.3], [0.5, 0.7, 1.0]]]],
        dtype=torch.float32,
    )

    output = layer(image)

    assert output.shape == (1, 3, 3, 3)


def test_hybrid_forward_requires_dataset_stats():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            }
        ],
        device="cpu",
        scale_mode="hybrid",
    )
    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="scale_mode='hybrid' requires dataset statistics"):
        layer(image)


def test_filter_spec_rejects_removed_output_mode():
    with pytest.raises(ValueError, match="output_mode was removed"):
        ConnectedFilterPreprocessingLayer(
            in_channels=1,
            filter_specs=[
                {
                    "tree_type": morphology.TreeType.MAX_TREE,
                    "attributes": (morphology.AttributeType.AREA,),
                    "output_mode": "tophat",
                }
            ],
            device="cpu",
            scale_mode="none",
        )


def test_save_params_includes_filter_specs_metadata(tmp_path):
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=1,
        filter_specs=[
            {
                "name": "area_compactness",
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (
                    morphology.AttributeType.AREA,
                    morphology.AttributeType.COMPACTNESS,
                ),
                "monotonicity_weight": 0.25,
            },
            {
                "name": "tos_boundary",
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": (morphology.AttributeGroup.BOUNDARY,),
                "valuation": CFPValuation.ALTITUDE_TOPHAT,
                "tos_interpolation": morphology.ToSInterpolation.Min8cMax4c,
                "preserve_root": True,
            },
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
                "valuation": CFPValuation.node_attribute(morphology.AttributeType.MEAN_LEVEL),
            },
        ],
        device="cpu",
        scale_mode="none",
        clamp=(-8, 10),
    )
    path = tmp_path / "params.pt"

    layer.export_params(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert "filter_specs" in payload
    assert payload["clamp"] == [-8.0, 10.0]
    assert payload["weight_contract"] == layer.get_weight_contract()
    assert payload["contracts"] == layer.get_contracts()
    assert payload["contracts"]["inference_contract"] == layer.get_inference_contract()
    assert payload["contracts"]["parameter_contract"]["weights"] == {
        "area_compactness": [2],
        "tos_boundary": [15],
        "spec_002": [1],
    }
    assert payload["contracts"]["training_contract"]["filter_specs"][0]["monotonicity_weight"] == pytest.approx(0.25)
    assert "monotonicity_weight" not in payload["weight_contract"]["filter_specs"][0]
    assert set(payload["weights"]) == {"area_compactness", "tos_boundary", "spec_002"}
    assert payload["filter_specs"][0]["key"] == "area_compactness"
    assert payload["filter_specs"][0]["name"] == "area_compactness"
    assert payload["filter_specs"][0]["tree_type"] == "max-tree"
    assert payload["filter_specs"][0]["attributes"] == ["AREA", "COMPACTNESS"]
    assert payload["filter_specs"][0]["valuation"]["kind"] == "altitude"
    assert payload["filter_specs"][0]["valuation"]["attribute"] == "ALTITUDE"
    assert payload["filter_specs"][0]["preserve_root"] is False
    assert payload["filter_specs"][0]["monotonicity_weight"] == pytest.approx(0.25)
    assert payload["filter_specs"][1]["key"] == "tos_boundary"
    assert payload["filter_specs"][1]["name"] == "tos_boundary"
    assert payload["filter_specs"][1]["tree_type"] == "tree-of-shapes"
    assert payload["filter_specs"][1]["valuation"]["kind"] == "altitude_tophat"
    assert payload["filter_specs"][1]["valuation"]["attribute"] == "ALTITUDE"
    assert payload["filter_specs"][1]["preserve_root"] is True
    assert payload["filter_specs"][1]["monotonicity_weight"] == 0.0
    assert payload["filter_specs"][1]["tos_interpolation"] == "Min8cMax4c"
    assert payload["filter_specs"][2]["key"] == "spec_002"
    assert payload["filter_specs"][2]["name"] == "spec_002"
    assert payload["filter_specs"][2]["valuation"]["kind"] == "node_attribute"
    assert payload["filter_specs"][2]["valuation"]["attribute"] == "MEAN_LEVEL"
    assert payload["filter_specs"][2]["preserve_root"] is False
    assert payload["filter_specs"][2]["monotonicity_weight"] == 0.0
    assert payload["config"]["clamp"] == [-8.0, 10.0]
    assert payload["config"]["filter_specs"][0]["name"] == "area_compactness"
    assert payload["config"]["filter_specs"][0]["monotonicity_weight"] == pytest.approx(0.25)
    assert payload["config"]["filter_specs"][1]["preserve_root"] is True
    assert payload["config"]["filter_specs"][1]["tos_interpolation"] == "Min8cMax4c"

    alias_path = tmp_path / "params_alias.pt"
    layer.save_params(alias_path)
    alias_payload = torch.load(alias_path, map_location="cpu", weights_only=True)
    assert alias_payload["weight_contract"] == payload["weight_contract"]


def test_get_config_and_from_config_reconstruct_layer_contract():
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=2,
        filter_specs=[
            {
                "name": "variance_max",
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (
                    morphology.AttributeType.AREA,
                    morphology.AttributeType.COMPACTNESS,
                ),
                "valuation": CFPValuation.node_attribute(morphology.AttributeType.VARIANCE_LEVEL),
            },
            {
                "name": "tos_tophat",
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": (morphology.AttributeGroup.BOUNDARY,),
                "valuation": CFPValuation.ALTITUDE_TOPHAT,
                "tos_interpolation": morphology.ToSInterpolation.Min8cMax4c,
                "tos_infinity_seed_row": 1,
                "tos_infinity_seed_col": 2,
            },
        ],
        device="cpu",
        scale_mode="none",
        eps=1e-5,
        beta_f=2.5,
        clamp=(-8, 10),
        hybrid_k=4.0,
        hybrid_floor_a=0.1,
    )

    restored = ConnectedFilterPreprocessingLayer.from_config(layer.get_config(), device="cpu")
    serializer = cfp.ConfigSerializer()

    assert serializer.layer_config(layer) == layer.get_config()
    assert serializer.inference_contract(layer) == layer.get_inference_contract()
    assert serializer.training_contract(layer) == layer.get_training_contract()
    assert serializer.parameter_contract(layer) == layer.get_parameter_contract()
    assert restored.get_config() == layer.get_config()
    assert restored.get_weight_contract() == layer.get_weight_contract()
    assert restored.get_inference_contract() == layer.get_inference_contract()
    assert restored.get_training_contract() == layer.get_training_contract()
    assert restored.get_parameter_contract() == layer.get_parameter_contract()
    assert restored.out_channels == layer.out_channels
    assert set(restored._weights) == {"variance_max", "tos_tophat"}


def test_forward_orders_outputs_by_input_channel_then_filter_spec(monkeypatch):
    layer = ConnectedFilterPreprocessingLayer(
        in_channels=2,
        filter_specs=[
            {
                "tree_type": morphology.TreeType.MAX_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            },
            {
                "tree_type": morphology.TreeType.MIN_TREE,
                "attributes": (morphology.AttributeType.AREA,),
            },
        ],
        device="cpu",
        scale_mode="none",
    )
    image = torch.tensor(
        [
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[3.0, 3.0], [3.0, 3.0]],
            ]
        ],
        dtype=torch.float32,
    )

    def fake_compute_tree_payload(img_np, tree_key, *, update_stats):
        return {
            "info": {"channel_marker": int(img_np[0, 0])},
            "norm_attrs": {},
            "valuation_increments": {},
        }

    def fake_apply_spec(spec, info, norm_attrs, valuation_increments, beta_f):
        value = 10 * info["channel_marker"] + spec.index
        return torch.full((2, 2), value, dtype=next(layer.parameters()).dtype)

    monkeypatch.setattr(layer, "_compute_tree_payload", fake_compute_tree_payload)
    monkeypatch.setattr(layer, "_apply_spec", fake_apply_spec)

    output = layer(image)

    assert output.shape == (1, 4, 2, 2)
    assert torch.equal(output[0, 0], torch.full((2, 2), 20.0))
    assert torch.equal(output[0, 1], torch.full((2, 2), 21.0))
    assert torch.equal(output[0, 2], torch.full((2, 2), 30.0))
    assert torch.equal(output[0, 3], torch.full((2, 2), 31.0))


def test_legacy_layer_preserves_old_attributes_spec_contract():
    layer = ConnectedFilterPreprocessingLayerLegacy(
        in_channels=1,
        attributes_spec=[
            (morphology.AttributeType.AREA,),
            (morphology.AttributeType.GRAY_HEIGHT,),
        ],
        tree_type="max-tree",
        device="cpu",
        scale_mode="none",
    )
    image = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)

    output = layer(image)

    assert layer.out_channels == 2
    assert output.shape == (1, 2, 2, 2)


def test_forward_rejects_non_batched_input_shape():
    layer = _single_area_layer()

    with pytest.raises(AssertionError, match="expected"):
        layer(torch.zeros((1, 3, 3), dtype=torch.float32))


def test_forward_rejects_wrong_channel_count():
    layer = _single_area_layer()

    with pytest.raises(AssertionError, match="in_channels=1"):
        layer(torch.zeros((1, 2, 3, 3), dtype=torch.float32))


def test_inspection_rejects_invalid_image_rank():
    layer = _single_area_layer()

    with pytest.raises(ValueError, match="img must be"):
        layer.inspect_training_sample(torch.zeros((1, 1, 3, 3), dtype=torch.float32))


def test_inspection_missing_cache_key_requires_build_if_missing():
    layer = _single_area_layer()
    image = torch.zeros((1, 3, 3), dtype=torch.float32)

    with pytest.raises(KeyError, match="Tree/attributes not found"):
        layer.inspect_training_sample(image, idx=42, build_if_missing=False)


def test_indexed_dataset_wrapper_rejects_scalar_samples():
    class ScalarDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.tensor(float(idx))

    wrapper = IndexedDatasetWrapper(ScalarDataset())

    with pytest.raises(ValueError, match="Dataset samples must be"):
        wrapper[0]


def test_deserialize_stats_rejects_unknown_attribute_key():
    with pytest.raises(ValueError, match="unknown serialized attribute key"):
        deserialize_ds_stats({"NOT_A_PUBLIC_ATTRIBUTE": {}}, torch.device("cpu"))
