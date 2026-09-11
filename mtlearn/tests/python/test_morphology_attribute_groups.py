import numpy as np
import pytest

from mtlearn import morphology


GROUPS = [("DIST_TRANSF", 29), ("DIST_TRANSF_EXACT", 29), ("FILLED_SHAPE", 12)]


def _square_image(tree_type, *, hole=False):
    image = np.zeros((7, 7), dtype=np.uint8)
    image[1:6, 1:6] = 200
    if hole:
        image[2, 2] = 0
    return 255 - image if tree_type == "min-tree" else image


@pytest.mark.parametrize("group_name, attribute_count", GROUPS)
@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree", "tree-of-shapes"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance_and_filled_shape_groups_match_scalar_attributes(
    group_name, attribute_count, tree_type, dtype
):
    tree = morphology.build_tree(_square_image(tree_type, hole=True), tree_type)
    group = getattr(morphology.AttributeGroup, group_name)
    attributes = morphology.expand_attribute_group(group)
    assert len(attributes) == attribute_count
    assert len(set(attributes)) == attribute_count
    assert set(attributes).issubset(morphology.expand_attribute_group(morphology.AttributeGroup.ALL))

    layout, values = morphology.compute_attributes(tree, [group], dtype=dtype)
    assert set(layout) == {attribute.name for attribute in attributes}
    assert list(layout.values()) == list(range(attribute_count))
    assert values.shape == (tree.num_internal_node_slots, attribute_count)
    assert values.dtype == np.dtype(dtype)
    assert np.isfinite(values).all()
    descriptions = morphology.describe_all_attributes()
    for attribute in attributes:
        assert descriptions[attribute.name] == morphology.describe_attribute(attribute)
        assert descriptions[attribute.name]
        single = morphology.compute_single_attribute(tree, attribute, dtype=dtype)
        np.testing.assert_array_equal(single, values[:, layout[attribute.name]])


def test_attribute_catalog_contains_all_distance_and_filled_shape_attributes():
    descriptions = morphology.describe_all_attributes()
    attributes = morphology.expand_attribute_group(morphology.AttributeGroup.ALL)
    assert len(attributes) == len(set(attributes)) == len(descriptions) == 132
    names = set()
    for group_name, _ in GROUPS:
        names.update(attribute.name for attribute in morphology.expand_attribute_group(
            getattr(morphology.AttributeGroup, group_name)
        ))
    assert len(names - {"MAX_DIST"}) == 69
    assert names.issubset(descriptions)


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("exact", [False, True])
def test_distance_attributes_match_square_contour_distances(tree_type, dtype, exact):
    tree = morphology.build_tree(_square_image(tree_type), tree_type)
    node = tree.smallest_node(3 * 7 + 3)
    suffix = "_EXACT" if exact else ""
    group = getattr(morphology.AttributeGroup, "DIST_TRANSF" + suffix)
    layout, values = morphology.compute_attributes(tree, [group], dtype=dtype)
    distances = np.array([0] * 16 + [1] * 8 + [2], dtype=np.float64)
    squared = distances**2
    probabilities = np.array([16, 8, 1], dtype=np.float64) / 25
    expected = {
        "MAX_DIST": 2,
        "MAX_SQUARED_DIST": 4,
        "DIST_SUM": distances.sum(),
        "DIST_MEAN": distances.mean(),
        "DIST_VARIANCE": distances.var(),
        "DIST_SQUARED_SUM": squared.sum(),
        "DIST_SQUARED_MEAN": squared.mean(),
        "DIST_SQUARED_VARIANCE": squared.var(),
        "DIST_RMS": np.sqrt(squared.mean()),
        "DIST_MEDIAN": 0,
        "DIST_MODE": 0,
        "DIST_Q25": 0,
        "DIST_Q75": 1,
        "DIST_Q90": 1,
        "DIST_ENTROPY": -(probabilities * np.log2(probabilities)).sum(),
        "DIST_POSITIVE_AREA": 9,
        "DIST_LEVEL_COUNT": 3,
        "MAX_DIST_CENTER_ROW": 3,
        "MAX_DIST_CENTER_COLUMN": 3,
        "MAX_DIST_PLATEAU_AREA": 1,
        "MAX_DIST_PLATEAU_CENTROID_ROW": 3,
        "MAX_DIST_PLATEAU_CENTROID_COLUMN": 3,
        "DIST_WEIGHTED_CENTROID_ROW": 3,
        "DIST_WEIGHTED_CENTROID_COLUMN": 3,
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(values[node, layout[name + suffix]], reference, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("tree_type", ["max-tree", "min-tree"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_filled_shape_attributes_measure_enclosed_region_and_hole(tree_type, dtype):
    tree = morphology.build_tree(_square_image(tree_type, hole=True), tree_type)
    node = tree.smallest_node(1 * 7 + 1)
    layout, values = morphology.compute_attributes(tree, [morphology.AttributeGroup.FILLED_SHAPE], dtype=dtype)
    area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA, dtype=dtype)
    assert area[node] == 24
    expected = {
        "FILLED_AREA": 25,
        "FILLED_CENTROID_ROW": 3,
        "FILLED_CENTROID_COLUMN": 3,
        "FILLED_LENGTH_MAJOR_AXIS": np.sqrt(8),
        "FILLED_LENGTH_MINOR_AXIS": np.sqrt(8),
        "FILLED_AXIS_ORIENTATION": 0,
        "FILLED_ECCENTRICITY": 1,
        "FILLED_INERTIA": 4 / 25,
        "HOLE_AREA_FRACTION": 1 / 25,
        "FILLED_CENTROID_DISPLACEMENT_NORMALIZED": np.sqrt(2) / (24 * 5),
        "FILLED_COMPACTNESS": 1 / (8 * np.pi),
        "FILLED_CIRCULARITY": 1,
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(values[node, layout[name]], reference, rtol=1e-6, atol=1e-7)
