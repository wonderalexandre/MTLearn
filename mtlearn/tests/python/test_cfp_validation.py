import pytest

import mtlearn

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import torch
except Exception as exc:  # pragma: no cover
    pytest.skip(f"PyTorch unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology
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
            },
            {
                "name": "tos_boundary",
                "tree_type": morphology.TreeType.TREE_OF_SHAPES,
                "attributes": (morphology.AttributeGroup.BOUNDARY,),
                "valuation": CFPValuation.ALTITUDE_TOPHAT,
                "tos_interpolation": morphology.ToSInterpolation.Min8cMax4c,
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
    assert set(payload["weights"]) == {"area_compactness", "tos_boundary", "spec_002"}
    assert payload["filter_specs"][0]["key"] == "area_compactness"
    assert payload["filter_specs"][0]["name"] == "area_compactness"
    assert payload["filter_specs"][0]["tree_type"] == "max-tree"
    assert payload["filter_specs"][0]["attributes"] == ["AREA", "COMPACTNESS"]
    assert payload["filter_specs"][0]["valuation"]["kind"] == "altitude"
    assert payload["filter_specs"][0]["valuation"]["attribute"] == "ALTITUDE"
    assert payload["filter_specs"][1]["key"] == "tos_boundary"
    assert payload["filter_specs"][1]["name"] == "tos_boundary"
    assert payload["filter_specs"][1]["tree_type"] == "tree-of-shapes"
    assert payload["filter_specs"][1]["valuation"]["kind"] == "altitude_tophat"
    assert payload["filter_specs"][1]["valuation"]["attribute"] == "ALTITUDE"
    assert payload["filter_specs"][1]["tos_interpolation"] == "Min8cMax4c"
    assert payload["filter_specs"][2]["key"] == "spec_002"
    assert payload["filter_specs"][2]["name"] == "spec_002"
    assert payload["filter_specs"][2]["valuation"]["kind"] == "node_attribute"
    assert payload["filter_specs"][2]["valuation"]["attribute"] == "MEAN_LEVEL"
    assert payload["config"]["clamp"] == [-8.0, 10.0]
    assert payload["config"]["filter_specs"][0]["name"] == "area_compactness"
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

    assert restored.get_config() == layer.get_config()
    assert restored.get_weight_contract() == layer.get_weight_contract()
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
