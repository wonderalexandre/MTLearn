import pytest

import mtlearn

if not getattr(mtlearn, "WITH_TORCH", False):
    pytest.skip("build has no LibTorch support", allow_module_level=True)

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    pytest.skip(f"NumPy unavailable: {exc}", allow_module_level=True)

from mtlearn import morphology

pytestmark = pytest.mark.integration


def _small_image():
    return np.array([[1, 2], [3, 4]], dtype=np.uint8)


def test_tree_constructors_return_public_facade_type():
    image = _small_image()

    trees = [
        morphology.create_max_tree(image),
        morphology.create_min_tree(image),
        morphology.create_tree_of_shapes(image),
    ]

    for tree in trees:
        assert morphology.is_tree(tree)
        assert isinstance(tree, morphology.WeightedMorphologicalTree)
        assert tree.numRows == image.shape[0]
        assert tree.numCols == image.shape[1]
        assert tree.numNodes > 0
        assert tree.numInternalNodeSlots >= tree.numNodes


def test_tree_of_shapes_facade_accepts_interpolation_options():
    image = _small_image()

    tree = morphology.create_tree_of_shapes(
        image,
        interpolation="min4c-max8c",
        infinity_seed_row=0,
        infinity_seed_col=0,
    )

    assert tree.treeType == 2
    assert tree.hasTreeOfShapesAdjacencyPolicy is True
    assert tree.getTreeOfShapesMinTreeAdjacencyRadius() == 1.0
    assert tree.getTreeOfShapesMaxTreeAdjacencyRadius() == 1.5

    enum_tree = morphology.build_tree(
        image,
        "tree-of-shapes",
        tos_interpolation=morphology.ToSInterpolation.Min8cMax4c,
    )

    assert enum_tree.treeType == 2
    assert enum_tree.getTreeOfShapesMinTreeAdjacencyRadius() == 1.5
    assert enum_tree.getTreeOfShapesMaxTreeAdjacencyRadius() == 1.0


def test_build_tree_rejects_unknown_tree_type():
    with pytest.raises(ValueError, match="unknown tree_type"):
        morphology.build_tree(_small_image(), "not-a-tree")


def test_tree_constructors_reject_non_2d_images():
    image = np.array([1, 2, 3], dtype=np.uint8)

    with pytest.raises(ValueError, match="2D uint8 array"):
        morphology.create_max_tree(image)


def test_compute_attributes_returns_sorted_index_and_expected_shape():
    tree = morphology.create_max_tree(_small_image())

    attr_index, attr_values = morphology.compute_attributes(
        tree,
        [
            morphology.AttributeType.AREA,
            morphology.AttributeType.COMPACTNESS,
            morphology.AttributeGroup.TREE_TOPOLOGY,
        ],
    )

    assert list(attr_index.values()) == sorted(attr_index.values())
    assert attr_values.shape[0] == tree.numInternalNodeSlots
    assert attr_values.shape[1] == len(attr_index)
    assert "AREA" in attr_index
    assert "COMPACTNESS" in attr_index


def test_compute_attributes_accepts_float_dtype():
    tree = morphology.create_max_tree(_small_image())
    attributes = [morphology.AttributeType.AREA, morphology.AttributeType.COMPACTNESS]

    attr_index32, attr_values32 = morphology.compute_attributes(
        tree,
        attributes,
        dtype=np.float32,
    )
    attr_index64, attr_values64 = morphology.compute_attributes(
        tree,
        attributes,
        dtype=np.float64,
    )
    _, attr_values64_from_string = morphology.compute_attributes(
        tree,
        attributes,
        dtype="float64",
    )

    assert attr_index64 == attr_index32
    assert attr_values32.dtype == np.float32
    assert attr_values64.dtype == np.float64
    assert attr_values64_from_string.dtype == np.float64
    assert np.allclose(attr_values32, attr_values64, equal_nan=True)


def test_compute_single_attribute_matches_tree_node_space():
    tree = morphology.create_max_tree(_small_image())

    area = morphology.compute_single_attribute(tree, morphology.AttributeType.AREA)

    assert area.shape == (tree.numInternalNodeSlots,)
    assert area.dtype == np.float32


def test_compute_single_attribute_accepts_float_dtype():
    tree = morphology.create_max_tree(_small_image())

    area32 = morphology.compute_single_attribute(
        tree,
        morphology.AttributeType.AREA,
        dtype=np.float32,
    )
    area64 = morphology.compute_single_attribute(
        tree,
        morphology.AttributeType.AREA,
        dtype=np.float64,
    )
    area64_from_python_float = morphology.compute_single_attribute(
        tree,
        morphology.AttributeType.AREA,
        dtype=float,
    )

    assert area32.dtype == np.float32
    assert area64.dtype == np.float64
    assert area64_from_python_float.dtype == np.float64
    assert np.allclose(area32, area64)


def test_compute_attribute_rejects_non_float_dtype():
    tree = morphology.create_max_tree(_small_image())

    with pytest.raises(ValueError, match="dtype must be np.float32 or np.float64"):
        morphology.compute_single_attribute(
            tree,
            morphology.AttributeType.AREA,
            dtype=np.int32,
        )


def test_attribute_descriptions_are_exposed_through_facade():
    area_description = morphology.describe_attribute(morphology.AttributeType.AREA)
    all_descriptions = morphology.describe_all_attributes()

    assert area_description.startswith("Area:")
    assert all_descriptions["AREA"] == area_description
    assert "MAX_DIST" in all_descriptions
    assert "CONTOUR_PERIMETER" in all_descriptions


def test_attribute_filter_validates_node_sized_inputs():
    tree = morphology.create_max_tree(_small_image())
    attribute_filter = morphology.create_attribute_filter(tree)

    with pytest.raises(ValueError, match="criterion must have length"):
        attribute_filter.filteringSubtractiveRule([True])

    with pytest.raises(ValueError, match="attr must have length"):
        attribute_filter.filteringMin(np.ones(1, dtype=np.float32), 1.0)


def test_attribute_filter_accepts_float64_attributes():
    tree = morphology.create_max_tree(_small_image())
    area = morphology.compute_single_attribute(
        tree,
        morphology.AttributeType.AREA,
        dtype=np.float64,
    )
    attribute_filter = morphology.create_attribute_filter(tree)

    filtered_min = attribute_filter.filteringMin(area, float(np.median(area)))
    filtered_max = attribute_filter.filteringMax(area, float(np.median(area)))

    assert filtered_min.shape == _small_image().shape
    assert filtered_max.shape == _small_image().shape
    assert filtered_min.dtype == np.uint8
    assert filtered_max.dtype == np.uint8


def test_attribute_filter_rejects_non_float_attribute_array():
    tree = morphology.create_max_tree(_small_image())
    attribute_filter = morphology.create_attribute_filter(tree)

    with pytest.raises(ValueError, match="attr must be a 1D np.float32 or np.float64 array"):
        attribute_filter.filteringMin(
            np.ones(tree.numInternalNodeSlots, dtype=np.int32),
            1.0,
        )


def test_removed_backend_symbols_are_not_reexported():
    removed_symbols = [
        "TreeStats",
        "make_tree_stats",
        "make_tree_tensor",
        "ConnectedFilterByJacobian",
        "ConnectedFilterByMorphologicalTree",
        "InfoTree",
    ]

    for name in removed_symbols:
        assert not hasattr(mtlearn, name)
        assert not hasattr(mtlearn._bindings, name)
