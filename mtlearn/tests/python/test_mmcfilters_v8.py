import numpy as np
import pytest

from mtlearn import morphology


NEW_ATTRIBUTES = {"AXIS_ORIENTATION_SIGNED", "FILLED_AXIS_ORIENTATION_SIGNED", "RAW_INERTIA"}


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree", "tree-of-shapes"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("hole", [False, True])
def test_signed_orientations_and_raw_inertia_match_pixel_geometry(tree_type, dtype, direction, hole):
    rows, columns = np.indices((7, 7))
    filled = (rows >= 1) & (rows <= 5) & (columns >= 1) & (columns <= 5) & (abs(rows - columns) <= 3)
    if direction < 0:
        filled = filled[:, ::-1]
    support = filled.copy()
    if hole:
        support[3, 2] = False
    image = np.where(support, 200, 0).astype(np.uint8)
    if tree_type == "min-tree":
        image = 255 - image
    tree = morphology.build_tree(image, tree_type)
    node = tree.smallest_node(int(np.flatnonzero(support)[0]))
    node_support = filled if tree_type == "tree-of-shapes" else support
    coordinates = np.argwhere(node_support).astype(np.float64)
    centered = coordinates - coordinates.mean(axis=0)
    expected_inertia = np.square(centered).sum()
    layout, values = morphology.compute(tree, sorted(NEW_ATTRIBUTES), dtype=dtype)
    assert values.dtype == dtype
    mu20 = np.square(centered[:, 1]).sum()
    mu02 = np.square(centered[:, 0]).sum()
    mu11 = np.prod(centered, axis=1).sum()
    orientation = np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))
    expected = {"AXIS_ORIENTATION_SIGNED": orientation,
                "FILLED_AXIS_ORIENTATION_SIGNED": direction * 45,
                "RAW_INERTIA": expected_inertia}
    for name, reference in expected.items():
        np.testing.assert_allclose(values[node, layout[name]], reference, rtol=1e-6, atol=1e-5)
        np.testing.assert_array_equal(morphology.compute(tree, name, dtype=dtype), values[:, layout[name]])
    unsigned = morphology.compute(tree, "AXIS_ORIENTATION", dtype=dtype)
    np.testing.assert_allclose(np.abs(values[:, layout["AXIS_ORIENTATION_SIGNED"]]), unsigned, atol=1e-5)
    moments = [morphology.compute(tree, name, dtype=dtype) for name in ("CENTRAL_MOMENT_20", "CENTRAL_MOMENT_02")]
    np.testing.assert_allclose(values[:, layout["RAW_INERTIA"]], sum(moments), rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("shape", [(1, 1), (3, 3)])
def test_isotropic_regions_have_zero_signed_orientation(shape):
    tree = morphology.create_max_tree(np.ones(shape, dtype=np.uint8))
    for name in NEW_ATTRIBUTES - {"RAW_INERTIA"}:
        np.testing.assert_array_equal(morphology.compute(tree, name), [0])
    if shape == (1, 1):
        np.testing.assert_array_equal(morphology.compute(tree, "RAW_INERTIA"), [0])


@pytest.mark.parametrize("group_name,count,added", [
    ("SHAPE", 50, {"AXIS_ORIENTATION_SIGNED", "RAW_INERTIA"}),
    ("MOMENTS", 23, {"AXIS_ORIENTATION_SIGNED", "RAW_INERTIA"}),
    ("FILLED_SHAPE", 13, {"FILLED_AXIS_ORIENTATION_SIGNED"}),
    ("ALL", 136, NEW_ATTRIBUTES),
])
def test_public_groups_expand_with_v8_members(group_name, count, added):
    group = getattr(morphology.AttributeGroup, group_name)
    members = morphology.expand_attribute_group(group)
    assert len(members) == len(set(members)) == count
    assert {member.name for member in members} & NEW_ATTRIBUTES == added
    tree = morphology.create_max_tree(np.arange(9, dtype=np.uint8).reshape(3, 3))
    layout, values = morphology.Attribute.compute(tree, group_name, dtype="float64")
    assert set(layout) == {member.name for member in members}
    assert values.shape == (tree.num_internal_node_slots, count)
    assert values.dtype == np.float64


def test_compute_dispatches_scalar_group_iterable_and_names():
    tree = morphology.create_max_tree(np.array([[0, 1], [2, 3]], dtype=np.uint8))
    expected = morphology.compute(tree, morphology.AttributeType.AREA)
    assert expected.ndim == 1
    for request in ("AREA", morphology.Attribute.AREA):
        np.testing.assert_array_equal(morphology.Attribute.compute(tree, request), expected)
    for request in (["AREA", "AREA"], iter([morphology.AttributeType.AREA])):
        layout, values = morphology.compute(tree, request)
        assert layout == {"AREA": 0}
        np.testing.assert_array_equal(values[:, 0], expected)
    layout, values = morphology.compute(tree, ["MOMENTS", "RAW_INERTIA", "GRAY_LEVEL"])
    assert len(layout) == 24
    assert set(layout) >= {"RAW_INERTIA", "GRAY_LEVEL"}
    assert values.shape == (tree.num_internal_node_slots, 24)
    assert morphology.compute(tree, "GRAY_LEVEL").ndim == 1
    assert all(hasattr(morphology, name) for name in morphology.__all__)
    assert morphology.Attribute.describe("RAW_INERTIA") == morphology.describe_attribute(morphology.AttributeType.RAW_INERTIA)
    with pytest.raises(TypeError):
        morphology.Attribute.describe("MOMENTS")
    for owner in (morphology, morphology.Attribute):
        assert not hasattr(owner, "compute_attributes")
        assert not hasattr(owner, "compute_single_attribute")
    with pytest.raises(TypeError):
        morphology.Attribute.compute(tree, "AREA", morphology.NodeIdSpace.MORPHOLOGICAL_TREE, np.float64)
    with pytest.raises(ValueError, match="Unknown"):
        morphology.compute(tree, "BOX_WIDTH")
    with pytest.raises(ValueError, match="Unknown"):
        morphology.compute(tree, ["area"])
    with pytest.raises(TypeError):
        morphology.compute(tree, [1])
    with pytest.raises(TypeError):
        morphology.compute(None, "AREA")


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rule", ["filtering_by_pruning_min_rule", "filtering_by_pruning_max_rule"])
def test_pruning_threshold_is_inclusive(tree_type, dtype, rule):
    image = np.zeros((5, 5), dtype=np.uint8)
    image[1:4, 1:4] = 100
    image[2, 2] = 200
    if tree_type == "min-tree":
        image = 255 - image
    tree = morphology.build_tree(image, tree_type)
    attribute = np.ones(tree.num_internal_node_slots, dtype=dtype)
    apply = getattr(morphology.AttributeFilters(tree), rule)
    np.testing.assert_array_equal(apply(attribute, 1), image)
    np.testing.assert_array_equal(apply(attribute, 1), apply(attribute >= 1))
    threshold = np.nextafter(dtype(1), dtype(2))
    np.testing.assert_array_equal(apply(attribute, threshold), apply(attribute >= threshold))
    assert not np.array_equal(apply(attribute, threshold), image)


@pytest.mark.parametrize("rule", ["filtering_by_pruning_min_rule", "filtering_by_pruning_max_rule",
                                  "filtering_by_direct_rule", "filtering_by_subtractive_rule"])
def test_masks_accept_readonly_noncanonical_bool_bytes_and_validate_shape(rule):
    image = np.array([[0, 40], [80, 200]], dtype=np.uint8)
    tree = morphology.create_max_tree(image)
    apply = getattr(morphology.AttributeFilters(tree), rule)
    mask = np.full(tree.num_internal_node_slots, 255, dtype=np.uint8).view(np.bool_)
    mask.flags.writeable = False
    np.testing.assert_array_equal(apply(mask), image)
    assert np.all(mask.view(np.uint8) == 255)
    with pytest.raises(TypeError):
        apply(mask.tolist())
    with pytest.raises(TypeError):
        apply(mask.view(np.uint8))
    with pytest.raises(ValueError):
        apply(mask[:-1])
    with pytest.raises(ValueError):
        apply(mask.reshape(1, -1))
    with pytest.raises(ValueError, match="contiguous"):
        apply(np.ones(mask.size * 2, dtype=np.bool_)[::2])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_soft_subtractive_scores_preserve_dtype_and_validate_range(dtype):
    image = np.array([[0, 40], [80, 200]], dtype=np.uint8)
    tree = morphology.create_max_tree(image)
    apply = morphology.AttributeFilters(tree).filtering_by_soft_subtractive_rule
    scores = np.ones(tree.num_internal_node_slots, dtype=dtype)
    scores.flags.writeable = False
    output = apply(scores)
    assert output.dtype == dtype
    np.testing.assert_array_equal(output, image)
    np.testing.assert_array_equal(apply(scores * 0.5), image * 0.5)
    for value in (-0.1, 1.1, np.inf, np.nan):
        invalid = scores.copy()
        invalid[-1] = value
        with pytest.raises(ValueError):
            apply(invalid)


@pytest.mark.parametrize("rule", ["filtering_by_pruning_min_rule", "filtering_by_pruning_max_rule",
                                  "filtering_by_direct_rule", "filtering_by_subtractive_rule",
                                  "filtering_by_soft_subtractive_rule", "filtering_by_extinction_rule"])
def test_filters_reject_stale_topology(rule):
    tree = morphology.create_max_tree(np.array([[0, 40], [80, 200]], dtype=np.uint8))
    filters = morphology.AttributeFilters(tree)
    tree.prune_node(tree.smallest_node(3))
    if rule == "filtering_by_extinction_rule":
        args = (morphology.compute(tree, "AREA"), morphology.ExtinctionSelectionPolicy.by_top_k(1))
    elif rule == "filtering_by_soft_subtractive_rule":
        args = (np.ones(tree.num_internal_node_slots, dtype=np.float32),)
    else:
        args = (np.ones(tree.num_internal_node_slots, dtype=np.bool_),)
    with pytest.raises(ValueError, match="topology changed"):
        getattr(filters, rule)(*args)


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree", "tree-of-shapes"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_expanded_groups_round_trip_and_train_in_cfp(tree_type, dtype):
    import torch
    from mtlearn.layers import ConnectedFilterPreprocessingLayer as Layer

    images = torch.randint(0, 256, (2, 1, 5, 5), dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(83))
    layer = Layer(1, [{"tree_type": tree_type, "attributes": [
        morphology.AttributeGroup.MOMENTS, morphology.AttributeGroup.FILLED_SHAPE]}],
        scale_mode="dataset_minmax01", attribute_dtype=dtype)
    layer.fit_stats([(images, torch.zeros(2))])
    config = layer.get_config()
    assert NEW_ATTRIBUTES.issubset(config["filter_specs"][0]["attributes"])
    restored = Layer.from_config(config)
    restored.load_state_dict(layer.state_dict())
    output = layer(images)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(restored(images), output, rtol=0, atol=0)
    (output / 255).square().mean().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
